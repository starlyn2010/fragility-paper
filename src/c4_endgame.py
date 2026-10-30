#!/usr/bin/env python3
"""c4_endgame.py — exact certificates on 7x6 Connect Four endgames.

Samples endgames with a fixed number of empty cells from random play, keeps
positions where the learner can still draw, solves each root exactly, trains a
history-dependent GRU against one deterministic optimal teacher, certifies it
against every optimal teacher *inside each root*, and repairs the violations
with certificate-guided repair.

The certificate is exact conditional on the sampled endgame suite: the
full-game 7x6 tree is not enumerated, but every teacher tie-break below each
root is.

Usage:
  python src/c4_endgame.py --roots 120 --empties 8 --seed 0 \
      --ckpt-out runs/c4_endgame_s0 --out data/c4_endgame_s0.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c4_game as g
from c4_repair import GRUPolicy, items_from, fit, n_optimal_mistakes

ROOT = pathlib.Path(__file__).resolve().parent.parent


def random_games(oracle, empties, n_roots, seed, root_value, max_games=200000):
    """Random play to N-empties; keep positions with the requested exact value."""
    rng = random.Random(seed)
    roots, seen = [], set()
    games = 0
    while len(roots) < n_roots and games < max_games:
        games += 1
        b, h = g.EMPTY, ()
        while not g.terminal(b) and len(h) < g.N - empties:
            c = rng.choice(g.legal(b))
            b = g.drop(b, c, g.turn(b))
            h = h + (c,)
        if g.terminal(b) or len(h) != g.N - empties:
            continue
        if b in seen:
            continue
        v = oracle.value(b)
        if v != root_value:
            continue
        seen.add(b)
        roots.append({"prefix": list(h), "board": list(b), "value": v})
    return roots, games


def root_items(root, oracle, order, rng, branch):
    """Learner-turn histories under a deterministic teacher inside the root."""
    out = []
    stack = [(tuple(root["board"]), tuple(root["prefix"]))]
    while stack:
        b, h = stack.pop()
        if g.terminal(b):
            continue
        if g.turn(b) == 1:
            out.append(h)
            acts = list(g.legal(b))
            if branch and len(acts) > branch:
                acts = rng.sample(acts, branch)
            for c in acts:
                stack.append((g.drop(b, c, 1), h + (c,)))
        else:
            c = g.teacher_action(b, oracle, order)
            stack.append((g.drop(b, c, 2), h + (c,)))
    return out


def certify(policy, oracle, roots):
    """Exact worst/best/uniform draw rates over all teacher tie-breaks."""
    per_root, violations = [], []
    for rid, root in enumerate(roots):
        b0, h0 = tuple(root["board"]), tuple(root["prefix"])

        def worst(b, h):
            if g.terminal(b):
                return 1.0 if g.winner(b) != 2 else 0.0
            if g.turn(b) == 1:
                a = policy.act(h, b)
                opts = oracle.optimal(b)
                if a not in opts:
                    violations.append({"root": rid, "history": list(h),
                                       "action": a, "optimal": list(opts)})
                return worst(g.drop(b, a, 1), h + (a,))
            return min(worst(g.drop(b, c, 2), h + (c,))
                       for c in oracle.optimal(b))

        def mean(b, h):
            if g.terminal(b):
                return 1.0 if g.winner(b) != 2 else 0.0
            if g.turn(b) == 1:
                a = policy.act(h, b)
                return mean(g.drop(b, a, 1), h + (a,))
            opts = oracle.optimal(b)
            return sum(mean(g.drop(b, c, 2), h + (c,)) for c in opts) / len(opts)

        per_root.append({"root": rid, "w": worst(b0, h0), "d": mean(b0, h0)})

    dedup = {}
    for v in violations:
        dedup.setdefault(tuple(v["history"]), v)
    violations = list(dedup.values())
    ws = [r["w"] for r in per_root]
    ds = [r["d"] for r in per_root]
    return {
        "worst_draw_rate": min(ws),
        "best_draw_rate": max(ws),
        "random_draw_rate_exact": sum(ds) / len(ds),
        "n_roots": len(roots),
        "n_roots_drawn_worst": sum(1 for w in ws if w == 1.0),
        "n_roots_drawn_mean": sum(1 for d in ds if d == 1.0),
        "violations": violations,
        "n_violations": len(violations),
        "per_root": per_root,
    }


def make_pool(roots, oracle, mode, seed, games=4):
    """Constrained learner histories for matched random controls."""
    rng = random.Random(seed)
    out, seen = [], set()
    for root in roots:
        for _ in range(games):
            b, h = tuple(root["board"]), tuple(root["prefix"])
            while not g.terminal(b):
                if g.turn(b) == 1:
                    opts = oracle.optimal(b)
                    if len(opts) < len(g.legal(b)) and h not in seen:
                        seen.add(h)
                        out.append(h)
                    acts = list(g.legal(b)) if mode == "all" else list(opts)
                    c = rng.choice(acts)
                else:
                    opts = oracle.optimal(b)
                    c = rng.choice(opts) if mode == "tree" else rng.choice(g.legal(b))
                b = g.drop(b, c, g.turn(b))
                h = h + (c,)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--empties", type=int, default=8)
    ap.add_argument("--roots", type=int, default=120)
    ap.add_argument("--roots-file", default="",
                    help="load a fixed root suite (JSON) instead of sampling one")
    ap.add_argument("--root-value", type=int, default=0,
                    help="exact value of every root from the learner's side "
                         "(0 draws, 1 learner wins)")
    ap.add_argument("--branch", type=int, default=3,
                    help="learner moves sampled per node in the training tree")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--agg-repeat", type=int, default=60)
    ap.add_argument("--stop", default="both", choices=["both", "agg", "replay"],
                    help="items that must be exactly optimal before a refit stops")
    ap.add_argument("--max-iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--control-seeds", default="1,2,3")
    ap.add_argument("--control-mode", default="all,tree")
    ap.add_argument("--control-games", type=int, default=3)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "c4_endgame.json"))
    ap.add_argument("--ckpt-out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    g.configure(args.rows, args.cols, args.win)
    t_start = time.time()
    oracle = g.Oracle()

    t0 = time.time()
    if args.roots_file:
        roots = json.loads(pathlib.Path(args.roots_file).read_text(encoding="utf-8"))
        games = 0
        print(f"roots loaded {len(roots)} from {args.roots_file}", flush=True)
    else:
        roots, games = random_games(oracle, args.empties, args.roots, args.seed,
                                    args.root_value)
    values = [r["value"] for r in roots]
    print(f"roots {len(roots)} from {games} random games "
          f"(value 0: {values.count(0)}, 1: {values.count(1)}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    order = tuple(range(g.COLS))
    rng = random.Random(args.seed + 1)
    train_hist, seen = [], set()
    for r in roots:
        for h in root_items(r, oracle, order, rng, args.branch):
            if h not in seen:
                seen.add(h)
                train_hist.append(h)
    replay = items_from(train_hist, oracle)
    print(f"replay histories {len(replay)} constrained "
          f"{len(items_from(train_hist, oracle, strict=True))}", flush=True)

    torch.manual_seed(args.seed)
    policy = GRUPolicy(args.hidden, args.emb)
    t0 = time.time()
    loss, eps = fit(policy, replay, args.epochs, args.lr, args.seed,
                    stop_items=replay)
    print(f"base fit loss {loss:.4f} epochs {eps} "
          f"mistakes {n_optimal_mistakes(policy, replay)} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    base = certify(policy, oracle, roots)
    print(f"base W {base['worst_draw_rate']:.0f} B {base['best_draw_rate']:.0f} "
          f"D {base['random_draw_rate_exact']:.4f} "
          f"roots {base['n_roots']} drawn {base['n_roots_drawn_worst']} "
          f"viol {base['n_violations']}", flush=True)

    agg, agg_set, log = [], set(), []
    policy0 = {k: v.clone() for k, v in policy.state_dict().items()}
    t0 = time.time()
    cert = base
    for it in range(args.max_iters):
        log.append({
            "round": it,
            "worst_draw_rate": cert["worst_draw_rate"],
            "best_draw_rate": cert["best_draw_rate"],
            "random_draw_rate_exact": cert["random_draw_rate_exact"],
            "n_violations": cert["n_violations"],
            "n_roots_drawn_worst": cert["n_roots_drawn_worst"],
            "aggregate": len(agg),
            "replay_mistakes": n_optimal_mistakes(policy, replay),
        })
        print(f"[repair it {it:2d}] W {cert['worst_draw_rate']:.0f} "
              f"B {cert['best_draw_rate']:.0f} "
              f"D {cert['random_draw_rate_exact']:.4f} "
              f"drawn {cert['n_roots_drawn_worst']}/{cert['n_roots']} "
              f"viol {cert['n_violations']:3d} agg {len(agg):3d} "
              f"rep_mis {log[-1]['replay_mistakes']}", flush=True)
        if cert["worst_draw_rate"] == 1.0:
            break
        new = [tuple(v["history"]) for v in cert["violations"]
               if tuple(v["history"]) not in agg_set]
        if not new:
            print("no new violating histories; stopping", flush=True)
            break
        for h in new:
            agg.append(h)
            agg_set.add(h)
        agg_items = items_from(agg, oracle)
        t1 = time.time()
        stop = (replay + agg_items if args.stop == "both"
                else agg_items if args.stop == "agg" else replay)
        l, e = fit(policy, replay + agg_items * args.agg_repeat,
                   args.epochs, args.lr, args.seed + it + 1,
                   stop_items=stop)
        print(f"    fit loss {l:.4f} epochs {e} "
              f"agg_mistakes {n_optimal_mistakes(policy, agg_items)} "
              f"[{time.time()-t1:.0f}s]", flush=True)
        cert = certify(policy, oracle, roots)
    repair_s = time.time() - t0
    final = certify(policy, oracle, roots)
    print(f"repair: W {final['worst_draw_rate']:.0f} "
          f"B {final['best_draw_rate']:.0f} "
          f"D {final['random_draw_rate_exact']:.4f} added {len(agg)} "
          f"[{repair_s:.0f}s]", flush=True)

    ckpt_path = ""
    if args.ckpt_out:
        p = pathlib.Path(args.ckpt_out)
        p.mkdir(parents=True, exist_ok=True)
        f = p / "c4_gru.pt"
        torch.save(policy.state_dict(), str(f))
        ckpt_path = str(f)
        print(f"saved {f}", flush=True)

    counts, running = [], 0
    prev = 0
    for e in log:
        running += e["aggregate"] - prev
        prev = e["aggregate"]
        counts.append(running)
    counts = counts[1:]

    controls = []
    if not args.no_controls:
        modes = [m.strip() for m in args.control_mode.split(",") if m.strip()]
        for mode in modes:
            pool = make_pool(roots, oracle, mode, seed=2026,
                             games=args.control_games)
            for cseed in [int(s) for s in args.control_seeds.split(",") if s.strip()]:
                torch.manual_seed(1000 + cseed)
                c_policy = GRUPolicy(args.hidden, args.emb)
                c_policy.load_state_dict(policy0)
                rngc = random.Random(cseed)
                c_agg, c_set = [], set()
                c_log = [{"round": 0,
                          "worst_draw_rate": base["worst_draw_rate"],
                          "best_draw_rate": base["best_draw_rate"],
                          "random_draw_rate_exact": base["random_draw_rate_exact"],
                          "n_violations": base["n_violations"],
                          "n_roots_drawn_worst": base["n_roots_drawn_worst"],
                          "added": 0}]
                for r, target in enumerate(counts):
                    need = target - len(c_agg)
                    candidates = [h for h in pool if h not in c_set]
                    for h in (rngc.sample(candidates, need)
                              if len(candidates) > need else candidates):
                        c_agg.append(h)
                        c_set.add(h)
                    c_items = items_from(c_agg, oracle)
                    stop = (replay + c_items if args.stop == "both"
                            else c_items if args.stop == "agg" else replay)
                    fit(c_policy, replay + c_items * args.agg_repeat,
                        args.epochs, args.lr, args.seed + 100 + r + cseed,
                        stop_items=stop)
                    cc = certify(c_policy, oracle, roots)
                    c_log.append({
                        "round": r + 1,
                        "worst_draw_rate": cc["worst_draw_rate"],
                        "best_draw_rate": cc["best_draw_rate"],
                        "random_draw_rate_exact": cc["random_draw_rate_exact"],
                        "n_violations": cc["n_violations"],
                        "n_roots_drawn_worst": cc["n_roots_drawn_worst"],
                        "added": len(c_agg),
                    })
                controls.append({"pool": mode, "seed": cseed, "log": c_log,
                                 "final": c_log[-1]})
                print(f"control {mode} s{cseed}: W {c_log[-1]['worst_draw_rate']:.0f} "
                      f"B {c_log[-1]['best_draw_rate']:.0f} "
                      f"D {c_log[-1]['random_draw_rate_exact']:.4f} "
                      f"added {c_log[-1]['added']}", flush=True)

    out = {
        "args": vars(args),
        "environment": {"python": __import__("platform").python_version(),
                        "torch": torch.__version__},
        "game": {"rows": g.ROWS, "cols": g.COLS, "win": g.K,
                 "empties": args.empties, "roots": len(roots),
                 "root_value_0": values.count(0), "root_value_1": values.count(1)},
        "replay_histories": len(replay),
        "base": {"certificate": {k: v for k, v in base.items() if k != "per_root"},
                 "fit_loss": loss, "fit_epochs": eps},
        "repair": {"log": log,
                   "final": {k: v for k, v in final.items() if k != "per_root"},
                   "per_root": final["per_root"],
                   "aggregate": agg, "elapsed_s": repair_s,
                   "checkpoint": ckpt_path},
        "controls": controls,
        "elapsed_s": time.time() - t_start,
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    rp = (ROOT / "data" /
          f"c4_endgame_roots_{args.empties}_{len(roots)}_s{args.seed}.json")
    rp.write_text(json.dumps(roots), encoding="utf-8")
    print(f"wrote {p} and {rp}", flush=True)


if __name__ == "__main__":
    main()
