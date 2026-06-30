#!/usr/bin/env python3
"""
evaluate_cross.py — Matriz de evaluación cruzada para Paper B.
Carga modelos entrenados (ablation + human_like) y evalúa contra 5+ oponentes.

Oponentes:
  - Random (StochasticMinimax depth=0)
  - Shadow (EMA del agente)
  - M4 (Minimax depth=4)
  - M6 (Minimax depth=6)
  - BoundedRationalMinimax (depth_normal=2, depth_tension=4, noise_std=0.5)

Uso:
  python3 evaluate_cross.py                               # Evaluar todos los modelos disponibles
  python3 evaluate_cross.py --models ablation_logs/*.pth   # Evaluar modelos específicos
  python3 evaluate_cross.py --dry-run                      # Validar pipeline (10 juegos cada uno)
"""
import sys, os, json, glob, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from liquid_rl_trainer import (
    LiquidAgent, ShadowAgent, TicTacToeEnv, StochasticMinimax,
)
from human_like_agent import BoundedRationalMinimax
from ablation_ttt import AblationLiquidAgent

DEVICE = torch.device("cpu")
RESULTS_FILE = "cross_eval_results.json"
N_GAMES = 200  # partidas por par (modelo × oponente)


# ── Definición de oponentes ──
OPPONENTS = {
    "random": StochasticMinimax(depth=0),
    "m4": StochasticMinimax(depth=4, noise_prob=0.0),
    "m6": StochasticMinimax(depth=6, noise_prob=0.0),
    "bounded_rational": BoundedRationalMinimax(
        depth_normal=2, depth_tension=4, noise_std=0.5,
    ),
}


def evaluate_pair(env, agent, opponent, n: int) -> tuple:
    """Retorna (wr, dr, n_wins, n_draws) para el par agente-oponente."""
    is_shadow = type(opponent).__name__ == "ShadowAgent"
    wins = draws = 0
    for _ in range(n):
        hx = torch.zeros(1, agent.hidden_dim)
        obs = env.reset()
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            legal_mask = torch.zeros(agent.action_dim, dtype=torch.bool)
            for a in legal:
                legal_mask[a] = True
            if turn % 2 == 0:
                with torch.no_grad():
                    action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:
                if is_shadow:
                    opp_action, hx_opp = opponent.get_action(obs, hx, legal_mask=legal_mask)
                    hx = hx_opp
                else:
                    opp_action = opponent.get_action(env, legal)
                obs, _, done, _ = env.step(opp_action, player=env.HUMANO)
            turn += 1
            if done:
                break
        r = env.resultado()
        if r == 1:
            wins += 1
        elif r == 0:
            draws += 1
    return wins / n, draws / n, wins, draws


def load_model(
    model_path: str, hidden_dim: int = 300, cfg_id: str | None = None
) -> LiquidAgent:
    """Carga un modelo desde checkpoint."""
    if cfg_id:
        agent = AblationLiquidAgent(cfg_id, obs_dim=9, action_dim=9, hidden_dim=hidden_dim)
        agent.s12_enabled = cfg_id in ("A", "D")
    else:
        agent = LiquidAgent(obs_dim=9, action_dim=9, hidden_dim=hidden_dim, lora_rank=0)
    agent.s12_max_iters = 3
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    agent.load_state_dict(state, strict=False)
    agent.eval()
    return agent


def discover_models() -> list[dict]:
    """Descubre modelos entrenados en el directorio de trabajo."""
    models = []
    # Ablation models
    for f in sorted(glob.glob("ablation_logs/ablation_*.pth")):
        name = os.path.basename(f).replace(".pth", "")
        cfg_id = name.replace("ablation_", "")
        models.append({
            "path": f, "name": name, "cfg_id": cfg_id,
            "hidden_dim": 300 if cfg_id != "D" else 32,
        })
    # Human-like model
    for f in sorted(glob.glob("human_like_logs/best_model.pth")):
        models.append({
            "path": f, "name": "human_like", "cfg_id": None, "hidden_dim": 300,
        })
    # Standard ttt_model
    for f in sorted(glob.glob("ttt_model.pth")):
        models.append({
            "path": f, "name": "ttt_model", "cfg_id": None, "hidden_dim": 300,
        })
    return models


def main():
    parser = argparse.ArgumentParser(description="Cross-evaluation matrix")
    parser.add_argument("--models", nargs="*", help="Paths a modelos .pth")
    parser.add_argument("--games", type=int, default=N_GAMES, help="Partidas por par")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run: 10 juegos, 2 modelos")
    args = parser.parse_args()

    n = 10 if args.dry_run else args.games
    env = TicTacToeEnv()

    if args.models:
        models_cfg = []
        for p in args.models:
            models_cfg.append({"path": p, "name": os.path.basename(p), "cfg_id": None, "hidden_dim": 300})
    else:
        models_cfg = discover_models()

    if not models_cfg:
        print("  ⚠️  No se encontraron modelos. Entrena primero o especifica --models.")
        return

    if args.dry_run:
        models_cfg = models_cfg[:2]  # solo primeros 2 en dry-run
        print(f"  Dry-run: {len(models_cfg)} modelos × {len(OPPONENTS)} oponentes × {n} juegos")

    print(f"\n{'='*70}")
    print(f"  MATRIZ DE EVALUACIÓN CRUZADA")
    print(f"  Modelos: {len(models_cfg)}, Oponentes: {len(OPPONENTS)}, Juegos/par: {n}")
    print(f"{'='*70}")

    all_results = []
    t0 = time.time()

    for mc in models_cfg:
        print(f"\n  ── [{mc['name']}] ──")
        try:
            agent = load_model(mc["path"], hidden_dim=mc.get("hidden_dim", 300), cfg_id=mc.get("cfg_id"))
        except Exception as e:
            print(f"    ⚠️  Error cargando {mc['path']}: {e}")
            continue

        # Crear shadow del agente (necesario como oponente)
        shadow = ShadowAgent(agent, tau=0.0)
        opponents = {**OPPONENTS, "shadow": shadow}

        model_results = {"model": mc["name"], "path": mc["path"]}
        for opp_name, opp in opponents.items():
            wr, dr, nw, nd = evaluate_pair(env, agent, opp, n)
            model_results[f"wr_vs_{opp_name}"] = round(wr, 4)
            model_results[f"dr_vs_{opp_name}"] = round(dr, 4)
            print(f"    vs {opp_name:>20s}: WR={wr*100:6.2f}% DR={dr*100:6.2f}%  ({nw}/{nd}/{n})")

        all_results.append(model_results)

        # Guardar parcial
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Tabla resumen ──
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  RESUMEN — MATRIZ DE EVALUACIÓN CRUZADA")
    print(f"  Tiempo total: {elapsed:.0f}s")
    print(f"{'='*70}")

    opp_keys = list(OPPONENTS.keys()) + ["shadow"]
    header = f"  {'Modelo':20s}" + "".join(f"{'WR vs '+k:>14s}" for k in opp_keys)
    print(header)
    print(f"  {'-'*20}" + "".join(f"{'─'*14}" for _ in opp_keys))
    for r in all_results:
        row = f"  {r['model']:20s}"
        for k in opp_keys:
            wr = r.get(f"wr_vs_{k}", 0)
            row += f"{wr*100:>13.1f}%"
        print(row)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Resultados guardados en {RESULTS_FILE}")


if __name__ == "__main__":
    main()
