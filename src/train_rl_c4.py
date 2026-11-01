#!/usr/bin/env python3
"""train_rl_c4.py — an RL-trained Connect Four base policy, for the repair study.

Every other learner in this paper is distilled: it is shown the teacher's tree and
given exact labels, so it can only be as good as the imitation. That is enough to
show the fragility, but a reviewer can object that no reward-driven agent was ever
trained in the second domain. This script closes that gap.

What the learner sees: the history of column plays, and one scalar terminal reward
(+1 won, 0 drew, -1 lost). What it never sees: optimal sets, value targets, the
teacher's tree, or any label. The opponent is the same deterministic solver the
distilled learners imitate (an exact minimax oracle whose ties are broken by a fixed
column order), so the only difference from the published arms is the training signal.

Algorithm: REINFORCE with a learned state baseline (actor-critic, one update per game),
gamma = 1, on the distilled GRU's architecture (same embedding and hidden width; the
critic is one extra linear row). The exact certificate is recomputed after every logged
block, so W, B and Dbar are measured along the learning curve rather than inferred.

Usage:
  python src/train_rl_c4.py --games 5000 --seed 0 --out data/rl_c4_s0.json
  python src/c4_repair.py --init data/rl_c4_s0.pt --seed 0 --out data/c4_rl_repair_s0.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch
from torch.distributions import Categorical

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c4_game as g
import c4_repair as rp

ROOT = pathlib.Path(__file__).resolve().parent.parent


class RLAgent(torch.nn.Module):
    """The distilled policy's architecture plus one linear row for the baseline.

    Kept deliberately identical (embedding table, GRU, action head) so the RL arm and
    the distillation arm differ in their training signal and not in capacity.
    """

    def __init__(self, hidden: int = 32, emb: int = 8) -> None:
        super().__init__()
        self.start = g.COLS
        self.pad = g.COLS + 1
        self.emb = torch.nn.Embedding(g.COLS + 2, emb, padding_idx=self.pad)
        self.gru = torch.nn.GRU(emb, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, g.COLS)
        self.critic = torch.nn.Linear(hidden, 1)

    def _encode(self, history: tuple):
        toks = torch.tensor([[self.start] + list(history)], dtype=torch.long)
        lengths = torch.tensor([len(history) + 1])
        e = self.emb(toks)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]

    def _legal_logits(self, history: tuple, board: tuple):
        logits = self.head(self._encode(history))[0].clone()
        for c in range(g.COLS):
            if c not in g.legal(board):
                logits[c] = -1e9
        return logits

    def step(self, history: tuple, board: tuple):
        """Sample a legal column and return its log-prob and the baseline estimate."""
        z = self._encode(history)
        logits = self.head(z)[0].clone()
        for c in range(g.COLS):
            if c not in g.legal(board):
                logits[c] = -1e9
        dist = Categorical(logits=logits)
        a = int(dist.sample().item())
        return a, dist.log_prob(torch.tensor(a)), self.critic(z)[0, 0], dist.entropy().item()

    @torch.no_grad()
    def act(self, history: tuple, board: tuple | None = None) -> int:
        if board is None:
            board = g.board_of(history)
        return int(torch.argmax(self._legal_logits(history, board)).item())


def play_one(agent: RLAgent, oracle, order: tuple):
    """One game against the deterministic teacher: (terminal reward, per-step tensors)."""
    board, history, p = g.EMPTY, (), 1
    steps = []
    while not g.terminal(board):
        if p == 1:
            a, logp, v, ent = agent.step(history, board)
            steps.append((logp, v, ent))
        else:
            a = g.teacher_action(board, oracle, order)
        board, history, p = g.drop(board, a, p), history + (a,), 3 - p
    w = g.winner(board)
    return (1.0 if w == 1 else (0.0 if w == 0 else -1.0)), steps


def certify(agent: RLAgent, oracle, board_items):
    cert = g.certify(agent.act, oracle)
    orders = g.teacher_orders()
    twin = [g.evaluate_teacher(lambda h: agent.act(h), oracle, o) for o in orders]
    return cert, twin, rp.n_optimal_mistakes(agent, board_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--block", type=int, default=500, help="games between logged blocks")
    ap.add_argument("--cert-every", type=int, default=1, help="blocks between certificates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--emb", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--critic-coef", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data" / "rl_c4_s0.json"))
    ap.add_argument("--ckpt", default="", help="default: --out with the .pt suffix")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    g.configure(args.rows, args.cols, args.win)
    order = tuple(range(g.COLS))
    horizon = g.N

    oracle = g.Oracle()
    oracle.value(g.EMPTY)
    paths = oracle.all_boards()
    replay_hist = [h for b, h in paths.items()
                   if not g.terminal(b) and g.turn(b) == 1]
    board_items = rp.items_from(replay_hist, oracle)

    torch.manual_seed(args.seed)
    agent = RLAgent(args.hidden, args.emb)
    params = sum(p_.numel() for p_ in agent.parameters())
    distilled = sum(p_.numel() for p_ in rp.GRUPolicy(args.hidden, args.emb).parameters())
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    print(f"board {args.rows}x{args.cols} win {args.win}: horizon {horizon}, "
          f"{len(paths)} boards, {len(replay_hist)} learner boards, "
          f"RL params {params} vs distilled {distilled}", flush=True)

    cert, twin, mistakes = certify(agent, oracle, board_items)
    print(f"[     0] untrained init: W {cert['worst_draw_rate']:.2f} "
          f"B {cert['best_draw_rate']:.2f} "
          f"Dbar {100*cert['random_draw_rate_exact']:.1f}% "
          f"viol {cert['n_violations']} frontier {cert['n_frontier']} "
          f"boardmistakes {mistakes}", flush=True)

    t_start, log = time.time(), [{"games": 0, "mean_reward": None,
                                  "W": cert["worst_draw_rate"],
                                  "B": cert["best_draw_rate"],
                                  "Dbar": cert["random_draw_rate_exact"],
                                  "n_states": cert["n_states"],
                                  "n_frontier": cert["n_frontier"],
                                  "n_violations": cert["n_violations"],
                                  "board_mistakes": mistakes,
                                  "teacher_orders_drawn": sum(twin)}]
    seen = 0
    while seen < args.games:
        n = min(args.block, args.games - seen)
        t0 = time.time()
        rewards, losses, ents = [], [], []
        for _ in range(n):
            reward, steps = play_one(agent, oracle, order)
            rewards.append(reward)
            if not steps:
                continue
            loss = sum(-logp * (reward - v.detach().item())
                       + args.critic_coef * (v - reward) ** 2
                       for logp, v, ent in steps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()) / len(steps))
            ents.extend(e for _, _, e in steps)
        seen += n
        row = {"games": seen, "mean_reward": statistics.fmean(rewards),
               "draw_rate_vs_teacher": sum(r == 0 for r in rewards) / len(rewards),
               "win_rate_vs_teacher": sum(r > 0 for r in rewards) / len(rewards),
               "loss": statistics.fmean(losses) if losses else 0.0,
               "policy_entropy": statistics.fmean(ents) if ents else 0.0,
               "block_s": round(time.time() - t0, 1)}
        note = ""
        if len(log) % args.cert_every == 0 or seen == args.games:
            cert, twin, mistakes = certify(agent, oracle, board_items)
            row.update({"W": cert["worst_draw_rate"], "B": cert["best_draw_rate"],
                        "Dbar": cert["random_draw_rate_exact"],
                        "n_states": cert["n_states"], "n_frontier": cert["n_frontier"],
                        "n_violations": cert["n_violations"],
                        "board_mistakes": mistakes, "teacher_orders_drawn": sum(twin)})
            note = (f"W {row['W']:.2f} B {row['B']:.2f} Dbar {100*row['Dbar']:.1f}% "
                    f"viol {row['n_violations']} boardmistakes {mistakes} ")
        log.append(row)
        print(f"[{seen:6d}] reward {row['mean_reward']:+.3f} "
              f"draw {100*row['draw_rate_vs_teacher']:.1f}% "
              f"entropy {row['policy_entropy']:.2f} {note}[{row['block_s']:.0f}s]",
              flush=True)

    cert, twin, mistakes = certify(agent, oracle, board_items)
    ckpt = pathlib.Path(args.ckpt or (args.out.rsplit(".", 1)[0] + ".pt"))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": agent.state_dict(), "hidden": args.hidden,
                "emb": args.emb, "rows": args.rows, "cols": args.cols,
                "win": args.win, "seed": args.seed, "games": args.games}, ckpt)

    out = {"args": vars(args),
           "environment": {"python": sys.version.split()[0], "torch": torch.__version__},
           "game": {"rows": g.ROWS, "cols": g.COLS, "win": g.K, "horizon": horizon,
                    "reachable_boards": len(paths), "learner_boards": len(replay_hist),
                    "tie_orders": len(g.teacher_orders())},
           "parameters": {"rl": params, "distilled_same_size": distilled},
           "signal": "terminal reward only: no optimal sets, value targets or teacher tree",
           "base": {"cert": {k: cert[k] for k in
                             ("worst_draw_rate", "best_draw_rate",
                              "random_draw_rate_exact", "n_states", "n_frontier",
                              "n_violations")},
                    "board_mistakes": mistakes,
                    "teacher_orders_drawn": sum(twin),
                    "mean_draw_rate_vs_deterministic_orders": statistics.fmean(twin)},
           "log": log, "elapsed_s": round(time.time() - t_start, 1),
           "checkpoint": str(ckpt)}
    outpath = pathlib.Path(args.out)
    if not outpath.is_absolute():
        outpath = ROOT / outpath
    outpath.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {outpath} and {ckpt} [{time.time()-t_start:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
