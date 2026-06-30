#!/usr/bin/env python3
"""
tournament.py — Torneo round-robin entre modelos entrenados.
Cada par juega 200 partidas (100 como P1, 100 como P2).

Uso:
  python3 tournament.py                            # Todos los modelos disponibles
  python3 tournament.py --games 100                # 100 partidas por par
  python3 tournament.py --human-like human_like_logs/best_model.pth
"""
import sys, os, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from liquid_rl_trainer import LiquidAgent, TicTacToeEnv
from ablation_ttt import AblationLiquidAgent

DEVICE = torch.device("cpu")
RESULTS_FILE = "tournament_results.json"
N_GAMES = 200


class ModelOpponent:
    """Envuelve un LiquidAgent para usarlo como oponente via get_action(env, legal)."""
    def __init__(self, agent: LiquidAgent, name: str):
        self.agent = agent
        self.name = name
        self.hx = torch.zeros(1, agent.hidden_dim)

    def reset(self):
        self.hx = torch.zeros(1, self.agent.hidden_dim)

    def get_action(self, env, legal: list[int], pity: float = 0.0, depth: int | None = None) -> int:
        legal_mask = torch.zeros(self.agent.action_dim, dtype=torch.bool)
        for a in legal:
            legal_mask[a] = True
        obs = env._obs() if hasattr(env, "_obs") else np.array(_fallback_obs(env), dtype=np.float32)
        with torch.no_grad():
            action, self.hx = self.agent.get_action(
                obs, self.hx, legal_mask=legal_mask, deterministic=True
            )
        return action


def _fallback_obs(env):
    """Construye obs desde grid si _obs no existe."""
    grid = env.grid if hasattr(env, "grid") else []
    turn = env.turno if hasattr(env, "turno") else 1
    flat = [x for row in grid for x in row] if grid else [0] * 9
    return flat + [turn]


def load_models() -> list[ModelOpponent]:
    """Carga todos los modelos disponibles."""
    models = []

    # Ablation models
    ablation_cfgs = [
        ("A", 300, "ablation_logs/ablation_A.pth"),
        ("B", 300, "ablation_logs/ablation_B.pth"),
        ("C", 300, "ablation_logs/ablation_C.pth"),
        ("D", 32, "ablation_logs/ablation_D.pth"),
    ]
    for cfg_id, hidden_dim, path in ablation_cfgs:
        if not os.path.exists(path):
            print(f"  ⚠️  No encontrado: {path}")
            continue
        agent = AblationLiquidAgent(cfg_id, obs_dim=9, action_dim=9, hidden_dim=hidden_dim)
        agent.s12_enabled = cfg_id in ("A", "D")
        agent.s12_max_iters = 3
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        agent.load_state_dict(state, strict=False)
        agent.eval()
        models.append(ModelOpponent(agent, f"ablation_{cfg_id}"))

    # Human-like model
    hl_path = "human_like_logs/best_model.pth"
    if os.path.exists(hl_path):
        agent = LiquidAgent(obs_dim=9, action_dim=9, hidden_dim=300, lora_rank=0)
        agent.s12_enabled = False
        state = torch.load(hl_path, map_location=DEVICE, weights_only=True)
        agent.load_state_dict(state, strict=False)
        agent.eval()
        models.append(ModelOpponent(agent, "human_like"))

    return models


def play_match(env, model_a: ModelOpponent, model_b: ModelOpponent, n: int) -> tuple[int, int, int]:
    """Juega n partidas entre model_a (P1) y model_b (P2). Alterna quien empieza."""
    wins_a = 0
    wins_b = 0
    draws = 0

    for gidx in range(n):
        # Alternar primer turno entre modelos
        if gidx % 2 == 0:
            p1, p2 = model_a, model_b
            p1_id, p2_id = 1, -1
        else:
            p1, p2 = model_b, model_a
            p1_id, p2_id = 1, -1

        p1.reset()
        p2.reset()
        env.reset()
        done = False
        turn = 0

        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            if turn % 2 == 0:
                action = p1.get_action(env, legal)
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:
                action = p2.get_action(env, legal)
                obs, _, done, _ = env.step(action, player=env.HUMANO)
            turn += 1

        r = env.resultado()
        if r == p1_id:
            if gidx % 2 == 0:
                wins_a += 1
            else:
                wins_b += 1
        elif r == p2_id:
            if gidx % 2 == 0:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1

    return wins_a, wins_b, draws


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament")
    parser.add_argument("--games", type=int, default=N_GAMES, help="Partidas por par")
    parser.add_argument("--human-like", type=str, default=None,
                        help="Ruta al modelo human-like (default: human_like_logs/best_model.pth)")
    args = parser.parse_args()

    models = load_models()
    if len(models) < 2:
        print("  ⚠️  Se necesitan al menos 2 modelos.")
        return

    print(f"\n{'='*60}")
    print(f"  TORNEO ROUND-ROBIN — {len(models)} modelos × {args.games} partidas/par")
    print(f"{'='*60}")

    env = TicTacToeEnv()
    n_names = len(models)
    names = [m.name for m in models]

    # Inicializar matriz triangular
    matrix = {name: {n: None for n in names} for name in names}

    t0 = time.time()

    for i in range(n_names):
        for j in range(i + 1, n_names):
            wa, wb, d = play_match(env, models[i], models[j], args.games)
            wr_a = wa / args.games * 100
            wr_b = wb / args.games * 100
            dr = d / args.games * 100
            matrix[names[i]][names[j]] = round(wr_a, 1)
            matrix[names[j]][names[i]] = round(wr_b, 1)
            print(f"  {names[i]:>16s} vs {names[j]:<16s}  "
                  f"{wr_a:5.1f}% / {dr:5.1f}% / {wr_b:5.1f}%  "
                  f"(W{wa}/D{d}/W{wb})")

    elapsed = time.time() - t0

    # ── Matriz ──
    print(f"\n{'='*60}")
    print(f"  MATRIZ DE RESULTADOS (WIN RATE del modelo fila vs columna)")
    print(f"  Tiempo: {elapsed:.0f}s")
    print(f"{'='*60}")

    col_width = max(10, max(len(n) for n in names) + 1)

    header = f"  {'':>{col_width}}"
    for n in names:
        header += f"  {n:>{col_width}s}"
    print(header)

    for n in names:
        row = f"  {n:>{col_width}}"
        for m in names:
            if n == m:
                val = matrix[n][m]
                if val is None:
                    row += f"  {'-':>{col_width}s}"
            else:
                row += f"  {matrix[n][m]:>{col_width}.1f}"
        print(row)

    # ── Ranking ──
    print(f"\n{'─'*40}")
    print(f"  RANKING (WR promedio)")
    print(f"{'─'*40}")
    rankings = []
    for name in names:
        wrs = [v for k, v in matrix[name].items() if v is not None]
        avg_wr = sum(wrs) / len(wrs)
        rankings.append((avg_wr, name))
    rankings.sort(reverse=True)
    for rank, (wr, name) in enumerate(rankings, 1):
        print(f"  {rank}. {name:>16s}  {wr:.1f}%")

    # ── Guardar ──
    result = {
        "models": names,
        "matrix": matrix,
        "ranking": [{"rank": r, "name": n, "avg_wr": w} for r, (w, n) in enumerate(rankings, 1)],
        "games_per_pair": args.games,
        "time_seconds": round(elapsed),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ {RESULTS_FILE}")


if __name__ == "__main__":
    main()
