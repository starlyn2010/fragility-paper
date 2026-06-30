#!/usr/bin/env python3
"""
benchmark_act.py — Adaptive Computation Time Benchmark.
Mide latencia y win rate del CfC bajo 3 modos de cómputo:
  - S1 Pure: 1 paso de ODE (forward_step)
  - ACT: forward_adaptive con effort_gate (early stopping)
  - S2 Pure: N pasos de ODE sin parada temprana

Output: benchmark_act_results.json + opcional benchmark_act.png
"""
import os, sys, json, math, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

from liquid_rl_trainer import LiquidAgent, TicTacToeEnv, StochasticMinimax

DEVICE = torch.device("cpu")
CHECKPOINT = "ttt_model.pth"
HIDDEN_DIM = 300
OBS_DIM = 9
ACTION_DIM = 9
RESULTS_FILE = "benchmark_act_results.json"

# ─── Config ───
N_GAMES = 100
N_BURNIN = 5
S2_MAX_ITERS = 5
ACT_MAX_ITERS = 3


def load_agent(path: str) -> LiquidAgent:
    agent = LiquidAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)
    sd = torch.load(path, map_location="cpu")
    miss, unexp = agent.load_state_dict(sd, strict=False)
    print(f"  Agent loaded: {len(miss)} missing, {len(unexp)} unexpected")
    agent.eval()
    return agent


def forward_s1(agent: LiquidAgent, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0):
    """S1 Pure: 1 paso de ODE."""
    logits, value, h_new = agent.forward_step(obs, hx, dt=dt)
    return logits, value, h_new, 1


def forward_act(agent: LiquidAgent, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0):
    """ACT: cómputo adaptativo con effort_gate."""
    logits, value, h_new, effort, n_iters = agent.forward_adaptive(obs, hx, dt=dt, max_iters=ACT_MAX_ITERS)
    return logits, value, h_new, n_iters


def forward_s2(agent: LiquidAgent, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0, max_iters: int = S2_MAX_ITERS):
    """S2 Pure: N pasos secuenciales, sin parada temprana."""
    h = hx
    for i in range(max_iters):
        logits, value, h = agent.forward_step(obs, h, dt=dt)
    return logits, value, h, max_iters


def benchmark_mode(agent: LiquidAgent, mode_name: str, forward_fn, opponent="minimax", n_games: int = N_GAMES, n_burnin: int = N_BURNIN):
    """Benchmark un modo contra un oponente. Retorna métricas."""
    env = TicTacToeEnv()
    mm = StochasticMinimax(depth=6) if opponent == "minimax" else None

    latencies = []   # ms por inferencia
    n_iters_list = []
    wins = 0
    total_games = 0
    total_inferences = 0

    n_total = n_burnin + n_games

    for game_idx in range(n_total):
        hx = torch.zeros(1, HIDDEN_DIM)
        obs = env.reset()
        done = False
        game_latencies = []
        game_n_iters = []

        while not done:
            legal = env.legal_actions()
            legal_mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
            for a in legal:
                legal_mask[a] = True

            if not legal:
                break

            # Agent's turn (odd = agent = 2)
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                t0 = time.perf_counter_ns()
                logits, value, hx_new, n_iters = forward_fn(agent, obs_t, hx, dt=1.0)
                t1 = time.perf_counter_ns()
                lat_ns = t1 - t0
                logits = logits.masked_fill(~legal_mask, -float("inf"))
                action = logits.argmax(dim=-1).item()

            if game_idx >= n_burnin:
                game_latencies.append(lat_ns / 1e6)  # ms
                game_n_iters.append(n_iters)

            hx = hx_new
            obs, _, done, _ = env.step(action, player=env.AGENTE)
            if done:
                break

            # Opponent's turn (random or minimax)
            opp_legal = env.legal_actions()
            if not opp_legal:
                break
            if mm:
                opp_action = mm.get_action(env, opp_legal, pity=0.0)
            else:
                opp_action = np.random.choice(opp_legal)
            obs, _, done, _ = env.step(opp_action, player=env.HUMANO)

        r = env.resultado()
        if game_idx >= n_burnin:
            total_games += 1
            total_inferences += len(game_latencies)
            latencies.extend(game_latencies)
            n_iters_list.extend(game_n_iters)
            if r == 1:
                wins += 1

    wr = wins / total_games if total_games > 0 else 0.0
    lat_arr = np.array(latencies)
    n_iters_arr = np.array(n_iters_list)

    stats = {
        "mode": mode_name,
        "opponent": opponent,
        "n_games": total_games,
        "n_inferences": total_inferences,
        "wr": round(wr, 4),
        "latency_ms": {
            "mean": round(float(lat_arr.mean()), 4),
            "std": round(float(lat_arr.std()), 4),
            "p50": round(float(np.percentile(lat_arr, 50)), 4),
            "p95": round(float(np.percentile(lat_arr, 95)), 4),
            "p99": round(float(np.percentile(lat_arr, 99)), 4),
            "min": round(float(lat_arr.min()), 4),
            "max": round(float(lat_arr.max()), 4),
        },
        "n_iters": {
            "mean": round(float(n_iters_arr.mean()), 4),
            "std": round(float(n_iters_arr.std()), 4),
            "min": int(n_iters_arr.min()),
            "max": int(n_iters_arr.max()),
            "histogram": {
                str(k): int((n_iters_arr == k).sum())
                for k in sorted(set(n_iters_arr.tolist()))
            },
        },
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="ACT Latency Benchmark")
    parser.add_argument("--headless", action="store_true", help="No plots")
    args = parser.parse_args()

    print("=" * 56)
    print("  ACT BENCHMARK: Tic-Tac-Toe")
    print("=" * 56)
    print(f"  Model: {CHECKPOINT} (hidden_dim={HIDDEN_DIM})")
    print(f"  Games/mode: {N_GAMES} (+{N_BURNIN} burn-in)")
    print(f"  S1:  1 ODE step")
    print(f"  ACT: max_iters={ACT_MAX_ITERS}, effort_gate halting")
    print(f"  S2:  {S2_MAX_ITERS} ODE steps, no early stopping")
    print(f"{'=' * 56}")

    agent = load_agent(CHECKPOINT)
    agent.s12_enabled = True  # needed for forward_adaptive

    modes = [
        ("S1_pure", forward_s1),
        ("ACT", forward_act),
        ("S2_pure", forward_s2),
    ]

    all_results = []

    for mode_name, forward_fn in modes:
        print(f"\n  ▶ {mode_name}")
        for opp_name in ("random", "minimax"):
            print(f"    vs {opp_name}...", end=" ", flush=True)
            stats = benchmark_mode(agent, mode_name, forward_fn, opponent=opp_name)
            print(f"WR={stats['wr']*100:.1f}%  lat={stats['latency_ms']['mean']:.3f}±{stats['latency_ms']['std']:.3f}ms  n_iters={stats['n_iters']['mean']:.2f}")
            all_results.append(stats)

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Results: {RESULTS_FILE}")

    # Plot
    if not args.headless:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a2e")

            colors = {"S1_pure": "#00e676", "ACT": "#7c4dff", "S2_pure": "#ff6d00"}
            labels = {"S1_pure": "S1 Pure (1 step)", "ACT": "Adaptive (ACT)", "S2_pure": f"S2 Pure ({S2_MAX_ITERS} steps)"}
            markers = {"random": "o", "minimax": "s"}

            # Plot 1: WR vs Latency (scatter)
            for r in all_results:
                c = colors.get(r["mode"], "#aaa")
                m = markers.get(r["opponent"], "o")
                lbl = labels.get(r["mode"], r["mode"])
                ax1.scatter(r["latency_ms"]["mean"], r["wr"] * 100,
                            c=c, marker=m, s=120, zorder=5,
                            label=f"{lbl} vs {r['opponent']}")
                ax1.annotate(r["mode"].split("_")[0],
                             (r["latency_ms"]["mean"], r["wr"] * 100),
                             textcoords="offset points", xytext=(5, 5),
                             fontsize=7, color="#ccc")

            ax1.set_xlabel("Mean Latency (ms)", color="#ccc")
            ax1.set_ylabel("Win Rate (%)", color="#ccc")
            ax1.set_title("Precision vs Costo Computacional", color="#e0e0e0")
            ax1.set_facecolor("#16213e")
            ax1.tick_params(colors="#aaa")
            ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
            ax1.grid(alpha=0.15)

            # Plot 2: n_iters histogram (ACT only)
            for r in all_results:
                if r["mode"] == "ACT" and r["opponent"] == "minimax":
                    hist = r["n_iters"]["histogram"]
                    if hist:
                        bins = sorted(int(k) for k in hist)
                        vals = [hist[str(k)] for k in bins]
                        ax2.bar(bins, vals, color="#7c4dff", alpha=0.8, width=0.6)
                        ax2.set_xlabel("n_iters (ODE steps)", color="#ccc")
                        ax2.set_ylabel("Frequency", color="#ccc")
                        ax2.set_title("ACT: Distribution of ODE Steps vs Minimax", color="#e0e0e0")
                        ax2.set_facecolor("#16213e")
                        ax2.tick_params(colors="#aaa")
                        ax2.grid(alpha=0.15)

            plt.tight_layout()
            path = "benchmark_act.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Plot: {path}")
        except ImportError:
            print("  (matplotlib not available)")

    # Summary table
    print(f"\n{'='*56}")
    print(f"  SUMMARY")
    print(f"{'='*56}")
    print(f"  {'Mode':<12} {'Opponent':<12} {'WR':>8} {'Lat(ms)':>12} {'n_iters':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    for r in all_results:
        print(f"  {r['mode']:<12} {r['opponent']:<12} {r['wr']*100:>7.1f}% {r['latency_ms']['mean']:>9.3f}ms {r['n_iters']['mean']:>7.2f}")
    print()

    print(f"  Key finding: ACT latency should be between S1 and S2,")
    print(f"  while WR should approach S2's level. Efficiency = WR / latency.")


if __name__ == "__main__":
    main()
