#!/usr/bin/env python3
"""repair.py — Certificate-guided adversarial repair (CGR).

The exact certificate answers "can this policy lose against an equally strong
teacher?" but it also returns the only states that matter for that answer: the
policy-dependent frontier R(pi), i.e. every learner-turn history reachable while
the teacher tie-breaks adversarially inside the optimal set.  If pi is optimal
on R(pi) it cannot lose; if it is not, the failing histories are witnesses.

CGR closes the loop:

  1. certify pi exactly  ->  W, B, exact random-tie draw rate, frontier R(pi)
  2. if W = 1 stop (certificate proves robustness)
  3. aggregate every violating frontier history with an exact oracle target
  4. fine-tune the *same-size* model on replay + aggregate
  5. go to 1

This is dataset aggregation (DAgger-style) over the class of optimal teachers,
with an exact stopping test instead of a sampled estimate.  The control arm adds
the same number of uniformly random histories per iteration, so any gain is
attributable to *which* states the certificate selects, not to extra data.

Uso:
  python src/repair.py --config C --ckpt runs/fixed_C_s42/ablation_C.pth \
      --out data/repair_C42.json --ckpt-out runs/repair_C42_s0
  python src/repair.py --config C --ckpt runs/distill_hist_C_histall_s0/ablation_C.pth \
      --label C-hist-full --replay canonical --out data/repair_histfull.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_ttt import CONFIGS, AblationLiquidAgent  # noqa: E402
from distill_history import (  # noqa: E402
    evaluate, forward_sequence, group_by_length, prepare,
)
from history_probe import enumerate_histories, history_actions  # noqa: E402
from liquid_rl_trainer import TicTacToeEnv  # noqa: E402
from oracle import Oracle, env_from_board  # noqa: E402
from worst_case_eval import (  # noqa: E402
    EMPTY, apply_move, board_state, load_recurrent, make_teacher,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def certify(policy, teacher_optimal, oracle):
    """Exact W/B/random-tie certificate plus the reachable frontier."""

    @lru_cache(maxsize=None)
    def rec(board: tuple[int, ...], hist: tuple):
        done, result, pieces = board_state(board)
        if done:
            return result, result, (1.0 if result == 0 else 0.0)

        if pieces % 2 == 0:  # learner
            action = policy(board, hist)
            nxt = apply_move(board, action, TicTacToeEnv.AGENTE)
            return rec(nxt, hist + (board,))

        actions = teacher_optimal(board)
        worst, best, p_draw = 2, -2, 0.0
        for action in actions:
            nxt = apply_move(board, action, TicTacToeEnv.HUMANO)
            w, b, pd = rec(nxt, hist)
            worst = min(worst, w)
            best = max(best, b)
            p_draw += pd
        return worst, best, p_draw / len(actions)

    worst, best, p_draw = rec(EMPTY, ())

    # Walk the frontier once to record actions and violations.
    frontier: dict[tuple, dict] = {}
    stack = [(EMPTY, ())]
    seen = set()
    while stack:
        board, hist = stack.pop()
        if (board, hist) in seen:
            continue
        seen.add((board, hist))
        done, _result, pieces = board_state(board)
        if done:
            continue
        if pieces % 2 == 0:
            action = policy(board, hist)
            info = oracle.lookup(board)
            regret = float(max(0.0, info["best"] - info["values"][action]))
            key = hist + (board,)
            frontier[key] = {
                "history_len": len(key),
                "action": int(action),
                "optimal": [int(a) for a in info["optimal"]],
                "best": float(info["best"]),
                "regret": regret,
                "violation": bool(action not in info["optimal"]),
            }
            stack.append((apply_move(board, action, TicTacToeEnv.AGENTE),
                          hist + (board,)))
        else:
            for action in teacher_optimal(board):
                stack.append((apply_move(board, action, TicTacToeEnv.HUMANO), hist))

    violations = [dict(key=list(k), **v) for k, v in frontier.items() if v["violation"]]
    return {
        "worst_draw_rate": 1.0 if worst == 0 else 0.0,
        "best_draw_rate": 1.0 if best == 0 else 0.0,
        "worst_result": int(worst),
        "best_result": int(best),
        "random_draw_rate_exact": float(p_draw),
        "n_states": int(rec.cache_info().currsize),
        "n_frontier": len(frontier),
        "n_violations": len(violations),
        "violations": violations,
        "max_regret": max((v["regret"] for v in frontier.values()), default=0.0),
    }


def item_for(history, board, oracle):
    info = oracle.lookup(board)
    values = [0.0] * 9
    for a, v in info["values"].items():
        values[a] = float(v)
    return {
        "history": tuple(history),
        "optimal": info["optimal"],
        "best": info["best"],
        "values": values,
        "tie": len(info["optimal"]) >= 2,
    }


def fit(agent, groups, epochs, batch_size, lr, seed, stop_groups=None):
    """Train until the aggregate is fitted exactly, or for `epochs` epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 1000)
    total, n = 0.0, 0
    epochs_run = 0
    for _ in range(epochs):
        epochs_run += 1
        agent.train()
        for length in sorted(groups, reverse=True):
            g = groups[length]
            order = rng.permutation(g["hist"].shape[0])
            for start in range(0, len(order), batch_size):
                idx = torch.from_numpy(order[start:start + batch_size])
                hist = g["hist"][idx]
                target = g["target"][idx]
                best = g["best"][idx]
                logits, value = forward_sequence(agent, hist)
                legal = hist[:, -1].eq(0.0)
                logp = torch.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)
                loss = (-(target * logp).sum(dim=-1).mean()
                        + 0.25 * torch.nn.functional.mse_loss(value, best))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()) * len(idx)
                n += len(idx)
        agent.eval()
        if stop_groups:
            m = evaluate(agent, stop_groups)
            if m["mean_regret"] <= 1e-9 and m["optimal_action_rate"] >= 1 - 1e-9:
                break
    agent.eval()
    return total / max(n, 1), epochs_run


def policy_of(agent, histories):
    table = history_actions(agent, histories)

    def policy(board, hist):
        key = hist + (board,)
        if key not in table:
            raise KeyError(f"history not enumerated: {key}")
        return table[key]

    return policy


def build_groups(replay, aggregate, oracle, agg_repeat=1):
    repair_items = [item_for(v["key"], tuple(v["key"][-1]), oracle)
                    for v in aggregate.values()]
    items = list(replay) + repair_items * max(1, agg_repeat)
    return group_by_length(items), group_by_length(repair_items)


def dagger_histories(policy, teacher_optimal, rng, games):
    """Learner-turn histories visited by the current policy against random
    optimal teachers: the DAgger state distribution at this round."""
    out: list[tuple] = []
    for _ in range(games):
        board, hist = EMPTY, ()
        while True:
            done, _result, pieces = board_state(board)
            if done:
                break
            if pieces % 2 == 0:
                key = hist + (board,)
                out.append(key)
                board = apply_move(board, int(policy(board, hist)),
                                   TicTacToeEnv.AGENTE)
                hist = key
            else:
                opts = teacher_optimal(board)
                board = apply_move(board, int(opts[int(rng.integers(len(opts)))]),
                                   TicTacToeEnv.HUMANO)
    return out


def add_history(aggregate, key, oracle):
    if key in aggregate:
        return False
    info = oracle.lookup(key[-1])
    aggregate[key] = {
        "key": list(key),
        "action": None,
        "optimal": [int(a) for a in info["optimal"]],
        "best": float(info["best"]),
        "regret": 0.0,
        "violation": None,
    }
    return True


def optimal_tree_histories(oracle, teacher_optimal):
    """Every agent-turn history in which both players play an optimal action."""
    out: dict[tuple, tuple] = {}
    stack = [(EMPTY, ())]
    while stack:
        board, hist = stack.pop()
        done, _result, pieces = board_state(board)
        if done:
            continue
        if pieces % 2 == 0:
            key = hist + (board,)
            out.setdefault(key, board)
            for action in oracle.lookup(board)["optimal"]:
                stack.append((apply_move(board, action, TicTacToeEnv.AGENTE), key))
        else:
            for action in teacher_optimal(board):
                stack.append((apply_move(board, action, TicTacToeEnv.HUMANO), hist))
    return list(out)


def run_arm(agent, histories, teacher_optimal, oracle, replay, epochs, lr,
            seed, max_iters, mode, counts, all_groups=None, agg_repeat=60,
            pool=None, dagger_games=100):
    """mode: 'repair' (certificate violations), 'random' (pool histories),
    'dagger' (learner rollouts, matched budget) or 'dagger-full' (all visited)."""
    rng = np.random.default_rng(seed)
    aggregate: dict[tuple, dict] = {}
    log = []
    cert = None
    for it in range(max_iters):
        t0 = time.time()
        cert = certify(policy_of(agent, histories), teacher_optimal, oracle)
        entry = {
            "iter": it,
            "agree": cert["worst_draw_rate"],
            "best": cert["best_draw_rate"],
            "random_draw_rate_exact": cert["random_draw_rate_exact"],
            "n_states": cert["n_states"],
            "n_frontier": cert["n_frontier"],
            "n_violations": cert["n_violations"],
            "aggregate": len(aggregate),
        }
        print(f"[{mode} it {it:2d}] W {entry['agree']:.0f} "
              f"B {entry['best']:.0f} random {entry['random_draw_rate_exact']:.4f} "
              f"frontier {entry['n_frontier']:3d} viol {entry['n_violations']:2d} "
              f"agg {len(aggregate):3d}", flush=True)
        log.append(entry)
        if cert["worst_draw_rate"] == 1.0:
            break
        if mode == "dagger" and counts is not None and it >= len(counts):
            break

        k = len(cert["violations"])
        if mode == "repair":
            before_agg = len(aggregate)
            for v in cert["violations"]:
                key = tuple(v["key"])
                aggregate.setdefault(key, v)
            log[-1]["aggregate_added"] = len(aggregate) - before_agg
        elif mode in ("dagger", "dagger-full"):
            visited = dagger_histories(policy_of(agent, histories),
                                       teacher_optimal, rng, dagger_games)
            cands = [h for h in dict.fromkeys(visited) if h not in aggregate]
            if mode == "dagger-full":
                need = len(cands)
            else:
                need = max(0, (counts[it] if counts else 0) - len(aggregate))
            if need < len(cands):
                order = rng.permutation(len(cands))[:need]
                cands = [cands[int(i)] for i in order]
            for key in cands:
                add_history(aggregate, key, oracle)
            log[-1]["aggregate_added"] = len(cands)
            log[-1]["visited"] = len(set(visited))
            log[-1]["visited_new"] = len(cands)
        else:
            target_k = counts[it] if counts is not None else k
            keys = [h["history"] for h in histories] if pool is None else pool
            while len(aggregate) < target_k:
                key = tuple(keys[int(rng.integers(len(keys)))])
                if key in aggregate:
                    continue
                add_history(aggregate, key, oracle)

        groups, repair_groups = build_groups(replay, aggregate, oracle, agg_repeat)
        before = evaluate(agent, repair_groups) if repair_groups else None
        loss, epochs_run = fit(agent, groups, epochs, 512, lr, seed + it,
                               stop_groups=repair_groups)
        after = evaluate(agent, repair_groups) if repair_groups else None
        log[-1]["fit_loss"] = loss
        log[-1]["fit_epochs"] = epochs_run
        log[-1]["fit_train_items"] = sum(g["hist"].shape[0] for g in groups.values())
        log[-1]["agg_regret_before"] = before["mean_regret"] if before else None
        log[-1]["agg_regret_after"] = after["mean_regret"] if after else None
        log[-1]["fit_s"] = time.time() - t0
        print(f"    fit loss {loss:.4f} epochs {epochs_run} agg_regret "
              f"{(before or {}).get('mean_regret', float('nan')):.4f} -> "
              f"{(after or {}).get('mean_regret', float('nan')):.4f}", flush=True)

    # Final certificate on the returned model.
    final_policy = policy_of(agent, histories)
    final = certify(final_policy, teacher_optimal, oracle)
    if all_groups is not None:
        final["all_history"] = evaluate(agent, all_groups)
    return {
        "log": log,
        "final": final,
        "aggregate": [
            {"key": list(k), **{kk: vv for kk, vv in v.items() if kk != "key"}}
            for k, v in aggregate.items()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C", choices=["B", "C"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--replay", default="canonical",
                    help="canonical | none | integer number of random histories")
    ap.add_argument("--epochs", type=int, default=12,
                    help="max fine-tune epochs per iteration (stops when the aggregate fits)")
    ap.add_argument("--agg-repeat", type=int, default=60,
                    help="oversample factor for the repair aggregate")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--control-seeds", default="1,2,3",
                    help="random-aggregation control seeds (empty = off)")
    ap.add_argument("--control-mode", default="all,tree",
                    help="control pools: all (180k histories) and/or tree (optimal play)")
    ap.add_argument("--dagger-seeds", default="1,2,3",
                    help="DAgger control seeds (empty = off)")
    ap.add_argument("--dagger-mode", default="matched,full",
                    help="matched (CGR per-round counts) and/or full (all visited)")
    ap.add_argument("--dagger-games", type=int, default=100,
                    help="rollouts per round for the DAgger controls")
    ap.add_argument("--eval-all", action="store_true",
                    help="exact all-history regret before/after (slow)")
    ap.add_argument("--ckpt-out", default="")
    ap.add_argument("--out", default=str(ROOT / "data" / "repair.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = next(c for c in CONFIGS if c["id"] == args.config)
    hidden = args.hidden or cfg["hidden_dim"]
    t_start = time.time()

    histories = enumerate_histories()
    oracle = Oracle()
    teacher_optimal = make_teacher()
    print(f"histories={len(histories)} boards={len({h['board'] for h in histories})}",
          flush=True)

    if args.replay == "none":
        replay = []
    elif args.replay == "canonical":
        replay = prepare(histories, oracle, "canonical")
    else:
        n = int(args.replay)
        perm = np.random.default_rng(99).permutation(len(histories))
        replay = prepare([histories[i] for i in perm[:n]], oracle, "all")
    print(f"replay={len(replay)}", flush=True)

    all_groups = None
    if args.eval_all:
        all_groups = group_by_length(prepare(histories, oracle, "all"))

    base = AblationLiquidAgent(args.config, 9, 9, hidden)
    base.s12_enabled = cfg["act"]
    base.s12_max_iters = 3
    base.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    base.eval()

    base_policy = policy_of(base, histories)
    base_cert = certify(base_policy, teacher_optimal, oracle)
    print(f"base W {base_cert['worst_draw_rate']:.0f} "
          f"B {base_cert['best_draw_rate']:.0f} "
          f"random {base_cert['random_draw_rate_exact']:.4f} "
          f"frontier {base_cert['n_frontier']} "
          f"viol {base_cert['n_violations']}", flush=True)
    base_all = evaluate(base, all_groups) if all_groups is not None else None

    # Arm 1: certificate-guided repair.
    agent = AblationLiquidAgent(args.config, 9, 9, hidden)
    agent.s12_enabled = cfg["act"]
    agent.s12_max_iters = 3
    agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    agent.eval()
    t0 = time.time()
    repair = run_arm(agent, histories, teacher_optimal, oracle, replay,
                     args.epochs, args.lr, args.seed, args.max_iters, "repair",
                     None, all_groups, args.agg_repeat)
    repair_s = time.time() - t0
    repaired_params = int(sum(p.numel() for p in agent.parameters()))
    print(f"repair: W {repair['final']['worst_draw_rate']:.0f} "
          f"random {repair['final']['random_draw_rate_exact']:.4f} "
          f"iters {len(repair['log']) - 1} added {len(repair['aggregate'])} "
          f"[{repair_s:.0f}s]", flush=True)

    ckpt_out = ""
    if args.ckpt_out:
        d = pathlib.Path(args.ckpt_out)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"ablation_{args.config}.pth"
        torch.save(agent.state_dict(), str(p))
        ckpt_out = str(p)
        print(f"saved {p}", flush=True)

    # Arm 2: the same number of states, chosen without looking at violations.
    # "all" samples uniformly from the 180,361 histories; "tree" samples from the
    # optimal-play tree, i.e. the right distribution but without selection.
    added = [max(1, e.get("aggregate_added", 0)) for e in repair["log"][:-1]]
    counts, running = [], 0
    for a in added:
        running += a
        counts.append(running)
    tree = optimal_tree_histories(oracle, teacher_optimal)
    print(f"optimal-play tree: {len(tree)} agent histories", flush=True)
    controls = []
    pools = {"all": None, "tree": tree}
    for mode in [m.strip() for m in args.control_mode.split(",") if m.strip()]:
        for seed in [int(s) for s in args.control_seeds.split(",") if s.strip()]:
            c_agent = AblationLiquidAgent(args.config, 9, 9, hidden)
            c_agent.s12_enabled = cfg["act"]
            c_agent.s12_max_iters = 3
            c_agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
            c_agent.eval()
            t0 = time.time()
            ctrl = run_arm(c_agent, histories, teacher_optimal, oracle, replay,
                           args.epochs, args.lr, seed, len(counts), "random",
                           counts, all_groups, args.agg_repeat, pools[mode])
            controls.append({
                "pool": mode,
                "seed": seed,
                "log": ctrl["log"],
                "final": ctrl["final"],
                "aggregate": ctrl["aggregate"],
                "elapsed_s": time.time() - t0,
            })
            print(f"control {mode} s{seed}: W {ctrl['final']['worst_draw_rate']:.0f} "
                  f"B {ctrl['final']['best_draw_rate']:.0f} "
                  f"random {ctrl['final']['random_draw_rate_exact']:.4f}", flush=True)

    # Arm 3: DAgger on the learner's own rollout distribution.  "matched" uses
    # CGR's per-round counts; "full" adds every visited state, i.e. the standard
    # unconstrained DAgger budget.
    daggers = []
    modes = [m.strip() for m in args.dagger_mode.split(",") if m.strip()]
    seeds = [int(s) for s in args.dagger_seeds.split(",") if s.strip()]
    if modes and seeds:
        for mode in modes:
            arm = "dagger" if mode == "matched" else "dagger-full"
            for seed in seeds:
                d_agent = AblationLiquidAgent(args.config, 9, 9, hidden)
                d_agent.s12_enabled = cfg["act"]
                d_agent.s12_max_iters = 3
                d_agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
                d_agent.eval()
                t0 = time.time()
                dg = run_arm(d_agent, histories, teacher_optimal, oracle, replay,
                             args.epochs, args.lr, seed, len(counts) + 1, arm,
                             counts if arm == "dagger" else None,
                             all_groups, args.agg_repeat, None, args.dagger_games)
                daggers.append({
                    "pool": mode,
                    "seed": seed,
                    "log": dg["log"],
                    "final": dg["final"],
                    "aggregate": dg["aggregate"],
                    "elapsed_s": time.time() - t0,
                })
                print(f"dagger-{mode} s{seed}: W {dg['final']['worst_draw_rate']:.0f} "
                      f"B {dg['final']['best_draw_rate']:.0f} "
                      f"random {dg['final']['random_draw_rate_exact']:.4f} "
                      f"added {len(dg['aggregate'])}", flush=True)

    out = {
        "args": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "when_utc": datetime.now(timezone.utc).isoformat(),
        },
        "n_histories": len(histories),
        "n_boards": len({h["history"][-1] for h in histories}),
        "optimal_tree": len(tree),
        "params": repaired_params,
        "base": {"certificate": base_cert, "all_history": base_all,
                 "sha256_16": hashlib.sha256(
                     pathlib.Path(args.ckpt).read_bytes()).hexdigest()[:16]},
        "repair": {**repair, "elapsed_s": repair_s, "checkpoint": ckpt_out},
        "controls": controls,
        "dagger": daggers,
        "elapsed_s": time.time() - t_start,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Crudo: {p}", flush=True)


if __name__ == "__main__":
    main()
