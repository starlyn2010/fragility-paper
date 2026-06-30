#!/usr/bin/env python3
"""
transfer_connect4.py — Transfer Learning: TTT backbone → Connect 4.
Two-phase: Linear Probing (F1) → Fine-tuning (F2), plus baseline from scratch.

Usage:
  python3 transfer_connect4.py                      # full run
  python3 transfer_connect4.py --quick               # 500+1000 games
  python3 transfer_connect4.py --headless            # no plots
"""
import os, sys, json, time, argparse, math, copy
import multiprocessing as mp_orig
mp = mp_orig.get_context("spawn")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from liquid_rl_trainer import (
    LiquidAgent, compute_reinforce_loss, EpisodeBuffer
)
from connect4_env import Connect4Env, MinimaxC4

DEVICE = torch.device("cpu")
CHECKPOINT_TTT = "ttt_model.pth"
CHECKPOINT_C4 = "ttt_model_c4.pth"
RESULTS_FILE = "transfer_results.json"

OBS_DIM_C4 = 43  # 42 celdas + 1 turno (del env._obs())
ACTION_DIM_C4 = 7
HIDDEN_DIM = 300
BOTTLENECK = 9  # 42→9 projection

# ─── Phase config ───
PHASE1_GAMES = 1000   # Pure Linear Probing (frozen CfC)
PHASE2_GAMES = 2000   # Fine-tuning (unfrozen CfC)
LR_ADAPTER = 3e-4
LR_BACKBONE = 1e-4
ENTROPY_COEF = 0.05
GAMMA = 0.99
CLIP_NORM = 5.0
EVAL_INTERVAL = 50

# ─── Opponent config ───
MINIMAX_DEPTH_TRAIN = 3
MINIMAX_DEPTH_EVAL = 4
PITY_NOISE = 0.20

# ─── Metric constants ───
ENTROPY_MAX = 1.9459  # ln(7), entropy of uniform policy over 7 actions


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class TransferC4Agent(nn.Module):
    """
    Wraps a pre-trained LiquidAgent with adapters for Connect4.
    
    Forward: obs(43) → InputAdapter(43→9) → LayerNorm(9) → CfC(9→300) → PolicyHead(300→7)
    
    Phase 1: CfC frozen, only adapters train.
    Phase 2: CfC unfrozen, dual-LR (low for backbone, normal for adapters).
    """

    def __init__(self, pretrained_path: str):
        super().__init__()
        # Expose attributes that compute_reinforce_loss needs
        self.hidden_dim = HIDDEN_DIM
        self.action_dim = ACTION_DIM_C4
        self.s12_enabled = False
        self.obs_mean = torch.zeros(OBS_DIM_C4)
        self.obs_std = torch.ones(OBS_DIM_C4)

        # ── Load backbone from TTT checkpoint ──
        self.backbone = LiquidAgent(obs_dim=9, action_dim=9, hidden_dim=HIDDEN_DIM)
        sd = torch.load(pretrained_path, map_location="cpu")
        miss, unexp = self.backbone.load_state_dict(sd, strict=False)
        print(f"  Backbone loaded: {len(miss)} missing (expected), {len(unexp)} unexpected")
        # Overwrite normalization buffers to work with 43-dim Connect4 observations
        with torch.no_grad():
            self.backbone.obs_mean = torch.zeros(OBS_DIM_C4)
            self.backbone.obs_std = torch.ones(OBS_DIM_C4)
        self.backbone.eval()

        # ── Adapters ──
        self.input_adapter = nn.Linear(OBS_DIM_C4, BOTTLENECK)
        self.input_adapter.apply(xavier_init)
        self.input_norm = nn.LayerNorm(BOTTLENECK)
        self.policy_head_c4 = nn.Linear(HIDDEN_DIM, ACTION_DIM_C4)
        self.value_head_c4 = nn.Linear(HIDDEN_DIM, 1)

        # ── Optimizer param groups (rebuilt each phase) ──
        self._phase = 1
        self._build_optimizer()

    def _build_optimizer(self):
        """Build optimizer with appropriate param groups for current phase."""
        adapter_params = list(self.input_adapter.parameters()) \
                       + list(self.input_norm.parameters()) \
                       + list(self.policy_head_c4.parameters()) \
                       + list(self.value_head_c4.parameters())

        if self._phase == 1:
            # Frozen backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.optimizer = torch.optim.AdamW(adapter_params, lr=LR_ADAPTER)
            print(f"  Phase 1: {sum(p.numel() for p in adapter_params):,} adapter params (LR={LR_ADAPTER})")
        else:
            # Unfrozen backbone with lower LR
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.optimizer = torch.optim.AdamW([
                {"params": adapter_params, "lr": LR_ADAPTER},
                {"params": self.backbone.parameters(), "lr": LR_BACKBONE},
            ])
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  Phase 2: {trainable:,}/{total:,} params (adapter LR={LR_ADAPTER}, backbone LR={LR_BACKBONE})")

    def set_phase(self, phase: int):
        if phase != self._phase:
            self._phase = phase
            self._build_optimizer()

    def forward_step(self, obs: torch.Tensor, hx: torch.Tensor, dt: float = 1.0):
        """
        obs: (B, 43) Connect4 board (42 cells + turn)
        hx:  (B, 300) hidden state
        Returns: (logits, value, h_new)
        """
        obs_proj = self.input_adapter(obs)       # (B, 43) → (B, 9)
        obs_proj = self.input_norm(obs_proj)     # (B, 9)  estabilizado
        h_new = self.backbone.liquid(obs_proj, hx, dt=dt)  # (B, 300)
        logits = self.policy_head_c4(h_new)      # (B, 7)
        value = self.value_head_c4(h_new).squeeze(-1)  # (B,)
        return logits, value, h_new

    def get_action(self, obs: np.ndarray, hx: torch.Tensor,
                   legal_mask: torch.Tensor | None = None,
                   deterministic: bool = False) -> tuple[int, torch.Tensor]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits, value, h_new = self.forward_step(obs_t, hx)
            if legal_mask is not None:
                logits = logits.masked_fill(~legal_mask, -float("inf"))
            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                action = Categorical(logits=logits).sample().item()
        return action, h_new


def make_baseline_agent():
    """Create a from-scratch agent with identical architecture."""
    return TransferC4Agent(CHECKPOINT_TTT)


def play_episode(agent, env, deterministic=False, explore_eps=0.0):
    """Play one Connect4 game. Returns EpisodeBuffer."""
    buffer = EpisodeBuffer()
    hx = torch.zeros(1, HIDDEN_DIM)
    obs = env.reset()
    done = False
    minimax = MinimaxC4(depth=MINIMAX_DEPTH_TRAIN)
    while not done:
        legal = env.legal_actions()
        legal_mask = torch.zeros(ACTION_DIM_C4, dtype=torch.bool)
        for a in legal:
            legal_mask[a] = True
        if len(legal) == 0:
            break
        # Agent's turn
        if random.random() < explore_eps:
            action = random.choice(legal)
            hx = torch.zeros(1, HIDDEN_DIM)
        else:
            action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=deterministic)
        new_obs, _, done, _ = env.step(action, player=env.AGENTE)
        buffer.obs.append(obs)
        buffer.actions.append(action)
        buffer.rewards.append(0.0)
        buffer.legal.append(legal)
        obs = new_obs
        if done:
            break
        # Opponent's turn (Minimax D3 + pity)
        opp_legal = env.legal_actions()
        if not opp_legal:
            break
        opp_action = minimax.get_action(env, opp_legal, pity=PITY_NOISE)
        obs, _, done, _ = env.step(opp_action, player=env.HUMANO)
    # Final reward
    r = env.resultado()
    final_r = 1.0 if r == 1 else (-0.5 if r == -1 else 0.3)
    if buffer.rewards:
        buffer.rewards[-1] += final_r
    return buffer


def evaluate(agent, env, n=50, opponent="minimax", depth=MINIMAX_DEPTH_EVAL, pity=0.0):
    """Evaluate win/draw rate against minimax (default) or random."""
    wins = 0
    draws = 0
    mm = MinimaxC4(depth=depth) if opponent == "minimax" else None
    for _ in range(n):
        hx = torch.zeros(1, HIDDEN_DIM)
        obs = env.reset()
        done = False
        turn = 0
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            legal_mask = torch.zeros(ACTION_DIM_C4, dtype=torch.bool)
            for a in legal:
                legal_mask[a] = True
            if turn % 2 == 0:  # agent's turn
                action, hx = agent.get_action(obs, hx, legal_mask=legal_mask, deterministic=True)
                obs, _, done, _ = env.step(action, player=env.AGENTE)
            else:
                if mm:
                    action = mm.get_action(env, legal, pity=pity)
                else:
                    action = random.choice(legal)
                obs, _, done, _ = env.step(action, player=env.HUMANO)
            turn += 1
            if done:
                break
        r = env.resultado()
        if r == 1:
            wins += 1
        elif r == 0:
            draws += 1
    return wins / n, draws / n


import random  # needed above


def run_training(num_games_f1, num_games_f2, label, results_queue):
    """Run two-phase training, push metrics to queue."""
    try:
        agent = TransferC4Agent(CHECKPOINT_TTT)
        env = Connect4Env()
        t0 = time.time()
        metrics = []
        game_count = 0

        # Phase 1: Pure Linear Probing
        agent.set_phase(1)
        print(f"\n  [{label}] Phase 1: Linear Probing ({num_games_f1} games)", flush=True)
        for _ in range(num_games_f1):
            buffer = play_episode(agent, env, explore_eps=0.0)
            if buffer.length > 0:
                loss_dict = compute_reinforce_loss(
                    agent, buffer, GAMMA, ENTROPY_COEF, DEVICE,
                    gae_lambda=0.95, s12_lambda=0.0,
                )
                agent.optimizer.zero_grad()
                if loss_dict["total"].grad_fn is not None:
                    loss_dict["total"].backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_NORM)
                    agent.optimizer.step()
            game_count += 1
            if game_count % EVAL_INTERVAL == 0:
                wr, dr = evaluate(agent, env, n=30, depth=MINIMAX_DEPTH_TRAIN, pity=PITY_NOISE)
                gn = 0.0
                for p in agent.parameters():
                    if p.grad is not None:
                        gn += p.grad.norm(2).item() ** 2
                gn = math.sqrt(gn)
                ent = loss_dict.get("entropy", torch.tensor(0.0)).item()
                elapsed = time.time() - t0
                metrics.append({
                    "phase": 1, "game": game_count,
                    "wr": round(wr, 4), "dr": round(dr, 4),
                    "loss": round(loss_dict["total"].item(), 4),
                    "entropy": round(ent, 4), "entropy_rel": round(ent / ENTROPY_MAX, 4),
                    "grad_norm": round(gn, 4),
                    "time": round(elapsed, 1),
                })
                print(f"  [{label}] F1 g={game_count:4d} WR={wr*100:5.1f}% DR={dr*100:5.1f}% loss={loss_dict['total'].item():.3f} H_rel={ent/ENTROPY_MAX:.3f} ∥g∥={gn:.2f}", flush=True)

        # Phase 2: Fine-tuning
        agent.set_phase(2)
        print(f"\n  [{label}] Phase 2: Fine-tuning ({num_games_f2} games)", flush=True)
        for _ in range(num_games_f2):
            buffer = play_episode(agent, env, explore_eps=0.0)
            if buffer.length > 0:
                loss_dict = compute_reinforce_loss(
                    agent, buffer, GAMMA, ENTROPY_COEF, DEVICE,
                    gae_lambda=0.95, s12_lambda=0.0,
                )
                agent.optimizer.zero_grad()
                if loss_dict["total"].grad_fn is not None:
                    loss_dict["total"].backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_NORM)
                    agent.optimizer.step()
            game_count += 1
            if game_count % EVAL_INTERVAL == 0:
                wr, dr = evaluate(agent, env, n=30, depth=MINIMAX_DEPTH_TRAIN, pity=PITY_NOISE)
                gn = 0.0
                for p in agent.parameters():
                    if p.grad is not None:
                        gn += p.grad.norm(2).item() ** 2
                gn = math.sqrt(gn)
                ent = loss_dict.get("entropy", torch.tensor(0.0)).item()
                elapsed = time.time() - t0
                metrics.append({
                    "phase": 2, "game": game_count,
                    "wr": round(wr, 4), "dr": round(dr, 4),
                    "loss": round(loss_dict["total"].item(), 4),
                    "entropy": round(ent, 4), "entropy_rel": round(ent / ENTROPY_MAX, 4),
                    "grad_norm": round(gn, 4),
                    "time": round(elapsed, 1),
                })
                print(f"  [{label}] F2 g={game_count:4d} WR={wr*100:5.1f}% DR={dr*100:5.1f}% loss={loss_dict['total'].item():.3f} H_rel={ent/ENTROPY_MAX:.3f} ∥g∥={gn:.2f}", flush=True)

        elapsed = time.time() - t0
        # ── Save checkpoint ──
        cp_path = f"transfer_{label}_c4.pth"
        torch.save(agent.state_dict(), cp_path)
        print(f"  [{label}] Checkpoint saved: {cp_path}", flush=True)
        # ── Evaluate vs Minimax ──
        print(f"  [{label}] Evaluating vs Minimax depth 4...", flush=True)
        mm_wr, mm_dr = evaluate(agent, env, n=50, opponent="minimax")
        print(f"  [{label}] vs Minimax d4: WR={mm_wr*100:.1f}% DR={mm_dr*100:.1f}%", flush=True)
        metrics.append({
            "phase": 3, "game": game_count,
            "wr": round(mm_wr, 4), "dr": round(mm_dr, 4),
            "loss": 0.0, "entropy": 0.0, "entropy_rel": 0.0, "grad_norm": 0.0,
            "time": round(elapsed, 1),
            "note": "evaluacion_vs_minimax_d4"
        })
        results_queue.put({"label": label, "metrics": metrics, "total_time": round(elapsed, 1)})
        print(f"\n  [{label}] Done in {elapsed:.0f}s", flush=True)
    except Exception as e:
        print(f"\n  [{label}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        results_queue.put({"label": label, "metrics": [], "total_time": 0.0})


def run_baseline(num_games, results_queue):
    """Run from-scratch baseline (no pretrained weights)."""
    try:
        env = Connect4Env()
        agent = TransferC4Agent(CHECKPOINT_TTT)
        # Randomize backbone (overwrite pretrained weights)
        for p in agent.backbone.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        # Train all params from the start
        agent._phase = 2  # force full training mode
        agent._build_optimizer()
        t0 = time.time()
        metrics = []

        print(f"\n  [Baseline] Training from scratch ({num_games} games)", flush=True)
        for game in range(1, num_games + 1):
            buffer = play_episode(agent, env, explore_eps=0.0)
            if buffer.length > 0:
                loss_dict = compute_reinforce_loss(
                    agent, buffer, GAMMA, ENTROPY_COEF, DEVICE,
                    gae_lambda=0.95, s12_lambda=0.0,
                )
                agent.optimizer.zero_grad()
                if loss_dict["total"].grad_fn is not None:
                    loss_dict["total"].backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_NORM)
                    agent.optimizer.step()
            if game % EVAL_INTERVAL == 0:
                wr, dr = evaluate(agent, env, n=30, depth=MINIMAX_DEPTH_TRAIN, pity=PITY_NOISE)
                gn = 0.0
                for p in agent.parameters():
                    if p.grad is not None:
                        gn += p.grad.norm(2).item() ** 2
                gn = math.sqrt(gn)
                ent = loss_dict.get("entropy", torch.tensor(0.0)).item()
                elapsed = time.time() - t0
                metrics.append({
                    "phase": 0, "game": game,
                    "wr": round(wr, 4), "dr": round(dr, 4),
                    "loss": round(loss_dict["total"].item(), 4),
                    "entropy": round(ent, 4), "entropy_rel": round(ent / ENTROPY_MAX, 4),
                    "grad_norm": round(gn, 4),
                    "time": round(elapsed, 1),
                })
                print(f"  [Baseline] g={game:4d} WR={wr*100:5.1f}% DR={dr*100:5.1f}% loss={loss_dict['total'].item():.3f} H_rel={ent/ENTROPY_MAX:.3f} ∥g∥={gn:.2f}", flush=True)

        elapsed = time.time() - t0
        # ── Save checkpoint ──
        cp_path = "baseline_c4.pth"
        torch.save(agent.state_dict(), cp_path)
        print(f"  [Baseline] Checkpoint saved: {cp_path}", flush=True)
        # ── Evaluate vs Minimax ──
        print(f"  [Baseline] Evaluating vs Minimax depth 4...", flush=True)
        mm_wr, mm_dr = evaluate(agent, env, n=50, opponent="minimax")
        print(f"  [Baseline] vs Minimax d4: WR={mm_wr*100:.1f}% DR={mm_dr*100:.1f}%", flush=True)
        metrics.append({
            "phase": 3, "game": game,
            "wr": round(mm_wr, 4), "dr": round(mm_dr, 4),
            "loss": 0.0, "entropy": 0.0, "entropy_rel": 0.0, "grad_norm": 0.0,
            "time": round(elapsed, 1),
            "note": "evaluacion_vs_minimax_d4"
        })
        results_queue.put({"label": "baseline", "metrics": metrics, "total_time": round(elapsed, 1)})
        print(f"\n  [Baseline] Done in {elapsed:.0f}s", flush=True)
    except Exception as e:
        print(f"\n  [Baseline] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        results_queue.put({"label": "baseline", "metrics": [], "total_time": 0.0})


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning TTT→Connect4")
    parser.add_argument("--quick", action="store_true", help="500+1000 games")
    parser.add_argument("--headless", action="store_true", help="No plots")
    args = parser.parse_args()

    p1_games = 500 if args.quick else PHASE1_GAMES
    p2_games = 1000 if args.quick else PHASE2_GAMES
    total = p1_games + p2_games

    print(f"{'='*56}")
    print(f"  TRANSFER LEARNING: TTT backbone → Connect 4")
    print(f"{'='*56}")
    print(f"  Phase 1 (Linear Probing): {p1_games} games (frozen CfC)")
    print(f"  Phase 2 (Fine-tuning):    {p2_games} games (unfrozen CfC)")
    print(f"  Baseline (from scratch):  {total} games")
    print(f"  Opponent: Minimax D3 + pity {PITY_NOISE}")
    print(f"  Eval every {EVAL_INTERVAL} games (D3+pity), final vs D4 (pity=0)")
    print(f"  InputAdapter + LayerNorm(9) → CfC")
    print(f"{'='*56}\n")

    # ── Run transfer + baseline in parallel ──
    results_queue = mp.Queue()

    p_transfer = mp.Process(target=run_training,
                            args=(p1_games, p2_games, "transfer", results_queue))
    p_baseline = mp.Process(target=run_baseline,
                            args=(total, results_queue))

    p_transfer.start()
    p_baseline.start()
    p_transfer.join()
    p_baseline.join()

    # ── Collect results ──
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*56}")
    print(f"  RESULTS: {RESULTS_FILE}")
    print(f"{'='*56}")
    for r in results:
        label = r["label"]
        ms = r["metrics"]
        print(f"\n  {label.upper()} ({r['total_time']:.0f}s):")
        if ms:
            # Find final training eval (phase 1 or 2) and final minimax eval (phase 3)
            train_ms = [m for m in ms if m["phase"] in (0, 1, 2)]
            mm_ms = [m for m in ms if m["phase"] == 3]
            if train_ms:
                last = train_ms[-1]
                wr_final = last["wr"]
                loss_final = last["loss"]
                h_rel = last.get("entropy_rel", 0.0)
                wr_max = max(m["wr"] for m in train_ms)
                print(f"    WR vs D3+pity: {wr_final*100:.1f}%  (max: {wr_max*100:.1f}%)")
                print(f"    Loss: {loss_final:.4f}  H_rel: {h_rel:.3f}")
            if mm_ms:
                mm = mm_ms[-1]
                print(f"    vs D4 (pity=0): WR={mm['wr']*100:.1f}% DR={mm['dr']*100:.1f}%")

    # ── Plot if headless not set ──
    if not args.headless:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#1a1a2e")
            colors = {"transfer": "#7c4dff", "baseline": "#ff6d00"}
            labels = {"transfer": "Transfer (TTT→C4)", "baseline": "From Scratch"}

            for r in results:
                ms = r["metrics"]
                games = [m["game"] for m in ms]
                wrs = [m["wr"] * 100 for m in ms]
                losses = [m["loss"] for m in ms]
                ents = [m["entropy"] for m in ms]
                gnorms = [m["grad_norm"] for m in ms]
                c = colors.get(r["label"], "#aaa")
                lbl = labels.get(r["label"], r["label"])

                ax = axes[0, 0]
                ax.plot(games, wrs, color=c, lw=1.5, label=lbl)
                ax.set_ylabel("Win Rate (%)", color="#ccc")
                ax.set_title("Win Rate vs Minimax D3+Pity", color="#e0e0e0")

                ax = axes[0, 1]
                ax.plot(games, losses, color=c, lw=1.5, label=lbl)
                ax.set_ylabel("Loss", color="#ccc")
                ax.set_title("REINFORCE Loss", color="#e0e0e0")

                ax = axes[1, 0]
                ax.plot(games, ents, color=c, lw=1.5, label=lbl)
                ax.set_ylabel("Entropy", color="#ccc")
                ax.set_title("Policy Entropy", color="#e0e0e0")

                ax = axes[1, 1]
                ax.plot(games, gnorms, color=c, lw=1.5, label=lbl)
                ax.set_ylabel("Gradient Norm", color="#ccc")
                ax.set_title("Gradient Norm", color="#e0e0e0")

            for ax_row in axes:
                for ax in ax_row:
                    ax.set_facecolor("#16213e")
                    ax.tick_params(colors="#aaa")
                    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
                    ax.grid(alpha=0.15)

            plt.tight_layout()
            path = "transfer_c4_results.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"\n  ✓ Plot: {path}")
        except ImportError:
            print("  (matplotlib not available for plotting)")

    print(f"\n{'='*56}")
    print(f"  Done. Results in {RESULTS_FILE}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
