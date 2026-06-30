#!/usr/bin/env python3
"""
ablation_ttt.py — Ablation Study: Tic-Tac-Toe.
Ejecuta 4 configuraciones para aislar el impacto de cada componente:

  A: CfC(9→300) + ACT (control — modelo actual)
  B: CfC(9→300) + 3 iteraciones fijas (sin ACT)
  C: GRU(9→300) + 1 paso (discreta, sin tiempo continuo)
  D: CfC(9→32)  + ACT (capacidad latente reducida)

Cada config entrena N juegos y evalúa contra Random, Shadow, M6 y BoundedRationalMinimax.

Uso:
  python3 ablation_ttt.py                       # 5000 juegos por config (~3h)
  python3 ablation_ttt.py --quick               # 1000 juegos, solo A y B
  python3 ablation_ttt.py --games 2000 --all    # 2000 juegos, 4 configs
"""
import sys, os, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from liquid_rl_trainer import (
    LiquidAgent, LiquidCell, Trainer, TrainingConfig,
    TicTacToeEnv, StochasticMinimax, ShadowAgent,
    CurriculumManager, HealthChecker, compute_reinforce_loss,
)
from human_like_agent import BoundedRationalMinimax

DEVICE = torch.device("cpu")
RESULTS_FILE = "ablation_results.json"
LOGS_DIR = "ablation_logs"
N_EVAL = 100


CONFIGS = [
    {
        "id": "A", "label": "CfC(9-300) + ACT",
        "hidden_dim": 300, "backbone": "cfc", "act": True, "lora_rank": 0,
    },
    {
        "id": "B", "label": "CfC(9-300) + fijo 3iters",
        "hidden_dim": 300, "backbone": "cfc_fixed3", "act": False, "lora_rank": 0,
    },
    {
        "id": "C", "label": "GRU(9-300) 1 paso",
        "hidden_dim": 300, "backbone": "gru", "act": False, "lora_rank": 0,
    },
    {
        "id": "D", "label": "CfC(9-32) + ACT",
        "hidden_dim": 32, "backbone": "cfc", "act": True, "lora_rank": 0,
    },
]


class AblationLiquidAgent(LiquidAgent):
    """Extiende LiquidAgent para usar distintos backbones según config."""

    def __init__(self, cfg_id: str, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__(obs_dim, action_dim, hidden_dim, lora_rank=0)
        self.cfg_id = cfg_id

        if cfg_id == "C":  # GRU en vez de CfC
            self.gru_cell = nn.GRUCell(obs_dim, hidden_dim)
            self.liquid = None
        elif cfg_id == "D":  # CfC con hidden_dim=32 (se pasa en constructor)
            self.liquid = LiquidCell(obs_dim, hidden_dim)

    def forward_step(self, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0):
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)

        if self.cfg_id == "C":  # GRU — 1 paso
            h_new = self.gru_cell(obs_norm, hx)

        elif self.cfg_id == "B":  # CfC — 3 pasos fijos sin halting
            h = hx
            for _ in range(3):
                h = self.liquid(obs_norm, h, dt=dt)
            h_new = h

        else:  # A o D: CfC — 1 paso estándar (ACT se maneja en forward_adaptive)
            h_new = self.liquid(obs_norm, hx, dt=dt)

        logits = self.policy_head(h_new)
        value = self.value_head(h_new).squeeze(-1)
        return logits, value, h_new

    def forward_lm_step(self, token_id: torch.Tensor, hx: torch.Tensor, dt: float = 0.3):
        """Override para GRU (Config C) y CfC pequeño (Config D)."""
        emb = self.text_embed(token_id)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        if self.lora_proj is not None:
            emb = emb + self.lora_proj(emb)
        if self.cfg_id == "C":
            h_new = self.gru_cell(emb, hx)
        else:
            h_new = self.liquid(emb, hx, dt=dt)
        h_dropped = self.lang_drop(self.lang_norm(h_new))
        logits = self.lang_head(h_dropped)
        return logits, h_new


def evaluate_agent(agent, env, opponent, n=N_EVAL):
    """Evalúa agente contra un oponente. Retorna (wr, dr)."""
    from liquid_rl_trainer import ShadowAgent

    wins = draws = 0
    is_shadow = type(opponent).__name__ == "ShadowAgent"

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
            if turn % 2 == 0:  # agent
                with torch.no_grad():
                    action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:  # opponent
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
    return wins / n, draws / n


def run_config(cfg: dict, num_games: int) -> dict:
    """Entrena y evalúa una configuración de ablation."""
    label = cfg["label"]
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"  Ablation: {label}")
    print(f"  hidden_dim={cfg['hidden_dim']}, backbone={cfg['backbone']}, ACT={cfg['act']}")
    print(f"{'='*50}")

    # ── Config de entrenamiento ──
    config = TrainingConfig(
        hidden_dim=cfg["hidden_dim"],
        lora_rank=cfg["lora_rank"],
        s12_enabled=cfg["act"],
        s12_max_iters=3,
        seed=42,
        device="cpu",
        curriculum_warmup=100,
        phase1_games_4=num_games // 3,
        phase2_games_4=num_games // 3,
        eval_interval=50,
        eval_episodes=30,
        pity_init_4=0.20,
        pity_final_4=0.05,
    )

    env = TicTacToeEnv()
    trainer = Trainer(config, env)

    # ── Reemplazar agente con variante de ablation ──
    agent = AblationLiquidAgent(cfg["id"], config.obs_dim, config.action_dim, config.hidden_dim)
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    trainer.agent = agent
    trainer.shadow = ShadowAgent(agent, tau=config.tau)
    trainer.optimizer = torch.optim.AdamW(agent.parameters(), lr=config.phase0_lr, weight_decay=config.weight_decay)

    # ── Entrenar ──
    agent = trainer.train(num_games)
    elapsed = time.time() - t0

    # ── Evaluar contra diferentes oponentes ──
    print(f"\n  Evaluando {label}...")
    opponents = {
        "random": StochasticMinimax(depth=0),
        "shadow": trainer.shadow,
        "m6": StochasticMinimax(depth=6, noise_prob=0.0),
        "bounded_rational": BoundedRationalMinimax(depth_normal=2, depth_tension=4, noise_std=0.5),
    }
    eval_results = {}
    for opp_name, opponent in opponents.items():
        w, d = evaluate_agent(agent, env, opponent, n=N_EVAL)
        eval_results[f"wr_vs_{opp_name}"] = round(w, 4)
        eval_results[f"dr_vs_{opp_name}"] = round(d, 4)
        print(f"    vs {opp_name:>16}: WR={w*100:.1f}% DR={d*100:.1f}%")

    # ── Extraer métricas del entrenamiento ──
    stats = trainer.stats
    result = {
        "config": cfg["id"],
        "label": label,
        "hidden_dim": cfg["hidden_dim"],
        "backbone": cfg["backbone"],
        "act": cfg["act"],
        "games": num_games,
        "time_seconds": round(elapsed),
        "time_str": f"{int(elapsed//60)}m {int(elapsed%60)}s",
        "params_total": sum(p.numel() for p in agent.parameters()),
        "params_trainable": sum(p.numel() for p in agent.parameters() if p.requires_grad),
        "metrics": {
            "wr_shadow_final": round(stats["wr_shadow"][-1] * 100, 2) if stats["wr_shadow"] else 0,
            "wr_shadow_max": round(max(stats["wr_shadow"]) * 100, 2) if stats["wr_shadow"] else 0,
            "entropy_final": round(stats["entropy"][-1], 4) if stats["entropy"] else 0,
            "grad_norm_avg": round(np.mean(stats["grad_norm"]), 4) if stats["grad_norm"] else 0,
            **eval_results,
        },
    }

    # ── Guardar checkpoint ──
    os.makedirs(LOGS_DIR, exist_ok=True)
    model_path = os.path.join(LOGS_DIR, f"ablation_{cfg['id']}.pth")
    torch.save(agent.state_dict(), model_path)
    result["model_path"] = model_path

    print(f"\n  ✓ {label}: {result['time_str']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Ablation Study TTT")
    parser.add_argument("--games", type=int, default=5000, help="Juegos por config")
    parser.add_argument("--quick", action="store_true", help="1000 juegos, solo A y B")
    parser.add_argument("--all", action="store_true", help="Todas las configs (default: A,B)")
    args = parser.parse_args()

    num_games = 1000 if args.quick else args.games
    configs = [c for c in CONFIGS if c["id"] in ("A", "B")] if not args.all else CONFIGS
    if not args.quick and args.all:
        configs = CONFIGS

    print(f"\n  Ablation Study TTT")
    print(f"  Juegos por config: {num_games}")
    print(f"  Configs: {len(configs)} ({', '.join(c['id'] for c in configs)})")

    all_results = []
    t_total = time.time()

    for i, cfg in enumerate(configs):
        print(f"\n  ─── [{i+1}/{len(configs)}] {cfg['label']} ───")
        result = run_config(cfg, num_games)
        all_results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  ✓ Parcial guardado en {RESULTS_FILE}")

    # ── Reporte final ──
    elapsed = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"  REPORTE ABLATION STUDY")
    print(f"{'='*50}")
    print(f"  Tiempo total: {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"\n  {'Config':16s} {'hidden':6s} {'WR_shad':8s} {'WR_M6':8s} {'WR_BR':8s} {'Params':8s} {'Time':8s}")
    print(f"  {'-'*16} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['config']:16s} {r['hidden_dim']:<6d} "
              f"{m['wr_shadow_final']:<8.1f} {m['wr_vs_m6']*100:<8.1f} {m['wr_vs_bounded_rational']*100:<8.1f} "
              f"{r['params_total']:<8,} {r['time_str']:<8s}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Final: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
