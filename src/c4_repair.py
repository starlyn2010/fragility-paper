#!/usr/bin/env python3
"""c4_repair.py — certificate-guided repair on Connect Four 4x4.

Trains a small history-dependent GRU against one deterministic optimal teacher
on a 4x4 Connect Four board, certifies it exactly against *every* optimal
teacher, and repairs it by adding only the violating frontier histories.

Usage:
  python src/c4_repair.py --seed 0 --max-iters 12 \
      --ckpt-out runs/c4_canon_s0 --out data/c4_canon_s0.json

  # resume a run that was stopped between (or inside) rounds
  python src/c4_repair.py --seed 0 --max-iters 12 --resume runs/c4_canon_s0 \
      --out data/c4_canon_s0.json

`--ckpt-out` holds `state.pt` (weights, aggregate, log, base certificate and
the next round) and `c4_gru.pt`, rewritten atomically after every certificate
and every refit. A killed run resumes from the last completed checkpoint and
continues with the same refit seeds, so a resumed arm is the arm that would
have run.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn

import c4_game as g

ROOT = pathlib.Path(__file__).resolve().parent.parent


class GRUPolicy(nn.Module):
    def __init__(self, hidden: int = 32, emb: int = 8) -> None:
        super().__init__()
        self.start = g.COLS
        self.pad = g.COLS + 1
        self.emb = nn.Embedding(g.COLS + 2, emb, padding_idx=self.pad)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, g.COLS)

    def forward(self, tokens, lengths):
        e = self.emb(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])

    @torch.no_grad()
    def logits(self, history: tuple):
        toks = torch.tensor([[self.start] + list(history)], dtype=torch.long)
        lengths = torch.tensor([len(history) + 1])
        return self.forward(toks, lengths)[0]

    @torch.no_grad()
    def act(self, history: tuple, board: tuple | None = None) -> int:
        if board is None:
            board = g.board_of(history)
        lg = self.logits(history).clone()
        acts = g.legal(board)
        for c in range(g.COLS):
            if c not in acts:
                lg[c] = -1e9
        return int(torch.argmax(lg).item())


def items_from(histories, oracle, strict: bool = False):
    out = []
    for h in histories:
        b = g.board_of(h)
        if g.terminal(b) or g.turn(b) != 1:
            continue
        opts = oracle.optimal(b)
        acts = g.legal(b)
        if strict and len(opts) == len(acts):
            continue
        target = torch.zeros(g.COLS)
        for c in opts:
            target[c] = 1.0 / len(opts)
        mask = torch.zeros(g.COLS)
        for c in acts:
            mask[c] = 1.0
        out.append({"history": h, "board": b, "target": target, "mask": mask,
                    "optimal": set(opts)})
    return out


def _batched(items, batch_size):
    start, pad = g.COLS, g.COLS + 1
    toks, lens = [], []
    for it in items:
        seq = [start] + list(it["history"])
        toks.append(seq)
        lens.append(len(seq))
    order = sorted(range(len(items)), key=lambda i: -lens[i])
    for i in range(0, len(items), batch_size):
        idx = order[i:i + batch_size]
        maxlen = max(lens[j] for j in idx)
        t = torch.full((len(idx), maxlen), pad, dtype=torch.long)
        for r, j in enumerate(idx):
            t[r, :lens[j]] = torch.tensor(toks[j])
        yield idx, t, torch.tensor([lens[j] for j in idx], dtype=torch.long)


def fit(policy, items, epochs, lr, seed, stop_items=None, log=print):
    if not items:
        return 0.0, 0
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    policy.train()
    last = 0.0
    for ep in range(epochs):
        shuffled = items[:]
        rng.shuffle(shuffled)
        total, nb = 0.0, 0
        for idx, toks, lens in _batched(shuffled, 256):
            logits = policy(toks, lens)
            targets = torch.stack([shuffled[j]["target"] for j in idx])
            masks = torch.stack([shuffled[j]["mask"] for j in idx])
            logits = logits.masked_fill(masks == 0, -1e9)
            loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        last = total / max(1, nb)
        if stop_items is not None and all_optimal(policy, stop_items):
            return last, ep + 1
    return last, epochs


@torch.no_grad()
def all_optimal(policy, items) -> bool:
    policy.eval()
    for it in items:
        if policy.act(it["history"], it["board"]) not in it["optimal"]:
            return False
    return True


def n_optimal_mistakes(policy, items) -> int:
    return sum(1 for it in items
               if policy.act(it["history"], it["board"]) not in it["optimal"])


def _is_constrained(history, oracle) -> bool:
    b = g.board_of(history)
    if g.terminal(b) or g.turn(b) != 1:
        return False
    return len(oracle.optimal(b)) < len(g.legal(b))


def sample_pool(pool, k, rng):
    if k <= 0:
        return []
    if len(pool) <= k:
        return list(pool)
    return rng.sample(pool, k)


def dagger_histories(policy, oracle, rng, games):
    """Learner-turn histories visited by the current policy against random
    optimal teachers: the DAgger state distribution at this round."""
    out = []
    for _ in range(games):
        b, h = g.EMPTY, ()
        while not g.terminal(b):
            if g.turn(b) == 1:
                out.append(h)
                c = policy.act(h, b)
            else:
                opts = oracle.optimal(b)
                c = int(opts[rng.randrange(len(opts))])
            b = g.drop(b, c, g.turn(b))
            h = h + (c,)
    return out


def certify_report(policy, oracle):
    cert = g.certify(lambda h: policy.act(h), oracle)
    cert["violations"] = [
        {"history": v["history"], "action": v["action"], "optimal": list(v["optimal"])}
        for v in cert["violations"]
    ]
    return cert


def _atomic_save(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--replay", default="canonical",
                    choices=["canonical", "teacher"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--agg-repeat", type=int, default=60)
    ap.add_argument("--stop", default="both", choices=["both", "agg", "replay"],
                    help="items that must be exactly optimal before a refit stops")
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--control-seeds", default="1,2,3")
    ap.add_argument("--control-mode", default="all,tree")
    ap.add_argument("--dagger-seeds", default="1,2,3")
    ap.add_argument("--dagger-mode", default="dagger,dagger-full",
                    help="dagger (CGR per-round counts) and/or dagger-full (all visited)")
    ap.add_argument("--dagger-games", type=int, default=100)
    ap.add_argument("--no-dagger", action="store_true")
    ap.add_argument("--control-epochs", type=int, default=0,
                    help="per-round epoch cap for controls; 0 uses --epochs")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "c4_repair.json"))
    ap.add_argument("--init", default="",
                    help="start from this policy checkpoint instead of distilling the "
                         "replay set; used to repair an RL-trained base (train_rl_c4.py)")
    ap.add_argument("--resume", default="",
                    help="continue the arm checkpointed in this directory; the base "
                         "certificate, aggregate, log and round index come from state.pt")
    ap.add_argument("--ckpt-out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    g.configure(args.rows, args.cols, args.win)

    preset = None
    resume_dir = ""
    if args.resume:
        rp = pathlib.Path(args.resume)
        if not rp.is_absolute():
            rp = ROOT / rp
        if rp.is_dir():
            rp = rp / "state.pt"
        if not rp.exists():
            raise SystemExit(f"--resume {args.resume}: no state.pt to resume from")
        preset = torch.load(rp, map_location="cpu", weights_only=False)
        resume_dir = str(rp.parent)
    ckpt_dir = pathlib.Path(args.ckpt_out) if args.ckpt_out else (
        pathlib.Path(resume_dir) if resume_dir else None)

    t_start = time.time()
    oracle = g.Oracle()
    paths = oracle.all_boards()
    board_count = len(paths)
    tau0 = tuple(range(g.COLS))
    if args.replay == "canonical":
        replay_hist = [h for b, h in paths.items()
                       if not g.terminal(b) and g.turn(b) == 1]
    else:
        replay_hist = g.tree_histories(oracle, tau0)
    replay = items_from(replay_hist, oracle)
    replay_strict = items_from(replay_hist, oracle, strict=True)
    print(f"boards={board_count} replay_histories={len(replay_hist)} "
          f"replay_items={len(replay)} constrained={len(replay_strict)}", flush=True)

    config = {"rows": args.rows, "cols": args.cols, "win": args.win,
              "hidden": args.hidden, "seed": args.seed, "replay": args.replay,
              "epochs": args.epochs, "lr": args.lr, "agg_repeat": args.agg_repeat,
              "stop": args.stop, "init": args.init, "threads": args.threads}
    if preset is not None:
        saved = preset["config"]
        mismatch = {k: (saved.get(k), v) for k, v in config.items()
                    if saved.get(k) != v}
        if mismatch:
            raise SystemExit("--resume does not match this run: " + ", ".join(
                f"{k} saved {a!r} asked {b!r}" for k, (a, b) in mismatch.items()))
        if preset["boards"] != board_count or preset["replay_items"] != len(replay):
            raise SystemExit(f"--resume was written on {preset['boards']} boards and "
                             f"{preset['replay_items']} replay items; this run builds "
                             f"{board_count} and {len(replay)}")
        if preset["done"]:
            raise SystemExit(f"--resume {resume_dir} is a finished arm: it stopped at "
                             f"round {preset['next_round']} ({preset['stop_reason']})")
        if preset["pending_round"] is not None and preset["next_round"] >= args.max_iters:
            raise SystemExit(f"--resume {resume_dir} still owes the refit of round "
                             f"{preset['pending_round']}; --max-iters {args.max_iters} "
                             f"stops before it, so raise the cap")

    torch.manual_seed(args.seed)
    policy = GRUPolicy(args.hidden)
    t0 = time.time()
    prior_elapsed = 0.0
    start_round, pending, repair_s = 0, None, 0.0
    if preset is not None:
        policy.load_state_dict(preset["policy_state"])
        base_cert, twin = preset["base"]["certificate"], preset["base"]["twin"]
        loss, eps = preset["base"]["loss"], preset["base"]["epochs"]
        base_source = preset["base"]["source"]
        policy_state0 = preset["policy_state0"]
        agg = [tuple(h) for h in preset["agg"]]
        agg_set = set(agg)
        log = preset["log"]
        start_round = int(preset["next_round"])
        pending = preset["pending_round"]
        repair_s = float(preset["repair_s"])
        prior_elapsed = float(preset["elapsed_s"])
        print(f"resumed {ckpt_dir} at round {start_round} (aggregate {len(agg)}, "
              f"{len(log)} rounds logged, {repair_s/60:.1f} repair-minutes spent)"
              + (f"; round {pending}'s refit is still owed" if pending is not None else ""),
              flush=True)
    elif args.init:
        path = pathlib.Path(args.init)
        if not path.is_absolute():
            path = ROOT / path
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        state = {k: v for k, v in state.items() if not k.startswith("critic.")}
        want = {"rows": args.rows, "cols": args.cols, "win": args.win,
                "hidden": args.hidden, "emb": 8}
        mismatch = {k: (blob[k], v) for k, v in want.items()
                    if isinstance(blob, dict) and k in blob and blob[k] != v}
        if mismatch:
            raise SystemExit(f"checkpoint {path} does not match this run: "
                             + ", ".join(f"{k} saved {a} asked {b}"
                                         for k, (a, b) in mismatch.items()))
        policy.load_state_dict(state)
        loss, eps, base_source = float("nan"), 0, str(path)
    else:
        loss, eps = fit(policy, replay, args.epochs, args.lr, args.seed,
                       stop_items=replay)
        base_source = "distilled on the replay set"

    if preset is None:
        print(f"base [{base_source}] fit loss {loss:.4f} epochs {eps} "
              f"mistakes {n_optimal_mistakes(policy, replay)} "
              f"[{time.time()-t0:.0f}s]", flush=True)

        base_cert = certify_report(policy, oracle)
        orders = g.teacher_orders()
        twin = [g.evaluate_teacher(lambda h: policy.act(h), oracle, o) for o in orders]
        print(f"base W {base_cert['worst_draw_rate']:.0f} "
              f"B {base_cert['best_draw_rate']:.0f} "
              f"D {base_cert['random_draw_rate_exact']:.4f} "
              f"states {base_cert['n_states']} frontier {base_cert['n_frontier']} "
              f"viol {base_cert['n_violations']} | {len(orders)} teachers: mean "
              f"{100*sum(twin)/len(twin):.1f} median {100*statistics.median(twin):.1f}",
              flush=True)

        policy_state0 = {k: v.clone() for k, v in policy.state_dict().items()}
        agg: list[tuple] = []
        agg_set: set[tuple] = set()
        log = []
    prior_repair_s = repair_s
    t_repair = time.time()

    def checkpoint(next_round, pending_round=None, done=False, stop_reason=""):
        if ckpt_dir is None:
            return
        now = time.time()
        weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        _atomic_save(weights, ckpt_dir / "c4_gru.pt")
        _atomic_save({
            "format": 1,
            "config": config,
            "next_round": next_round,
            "pending_round": pending_round,
            "done": done,
            "stop_reason": stop_reason,
            "policy_state": weights,
            "policy_state0": policy_state0,
            "agg": agg,
            "log": log,
            "base": {"certificate": base_cert, "twin": twin, "loss": loss,
                     "epochs": eps, "source": base_source},
            "boards": board_count,
            "replay_items": len(replay),
            "repair_s": prior_repair_s + (now - t_repair),
            "elapsed_s": prior_elapsed + (now - t_start),
        }, ckpt_dir / "state.pt")
        print(f"checkpoint round {next_round}"
              + (f" (refit of {pending_round} owed)" if pending_round is not None else "")
              + f" -> {ckpt_dir / 'state.pt'}", flush=True)

    # ---- repair arm ----
    if preset is None:
        checkpoint(start_round)
    stop_reason = ""
    it = start_round - 1
    try:
        for it in range(start_round, args.max_iters):
            if pending == it:
                pending = None
            else:
                cert = certify_report(policy, oracle)
                agg_items = items_from(agg, oracle)
                log.append({
                    "round": it,
                    "worst_draw_rate": cert["worst_draw_rate"],
                    "best_draw_rate": cert["best_draw_rate"],
                    "random_draw_rate_exact": cert["random_draw_rate_exact"],
                    "n_states": cert["n_states"],
                    "n_frontier": cert["n_frontier"],
                    "n_violations": cert["n_violations"],
                    "aggregate": len(agg),
                    "aggregate_mistakes": n_optimal_mistakes(policy, agg_items) if agg else 0,
                    "replay_mistakes": n_optimal_mistakes(policy, replay),
                })
                print(f"[repair it {it:2d}] W {cert['worst_draw_rate']:.0f} "
                      f"B {cert['best_draw_rate']:.0f} "
                      f"D {cert['random_draw_rate_exact']:.4f} "
                      f"frontier {cert['n_frontier']:4d} viol {cert['n_violations']:4d} "
                      f"agg {len(agg):3d} mistakes {log[-1]['aggregate_mistakes']}",
                      flush=True)
                if cert["worst_draw_rate"] == 1.0:
                    stop_reason = "certified"
                    break
                new = [v["history"] for v in cert["violations"]
                       if tuple(v["history"]) not in agg_set]
                if not new:
                    print("no new violating histories; stopping", flush=True)
                    stop_reason = "no new violations"
                    break
                for h in new:
                    agg.append(tuple(h))
                    agg_set.add(tuple(h))
                pending = it
                checkpoint(it, pending_round=it)
            agg_items = items_from(agg, oracle)
            agg_rep = agg_items * args.agg_repeat
            train_items = replay + agg_rep
            t1 = time.time()
            stop = (replay + agg_items if args.stop == "both"
                    else agg_items if args.stop == "agg" else replay)
            l, e = fit(policy, train_items, args.epochs, args.lr,
                       args.seed + it + 1, stop_items=stop)
            print(f"    fit loss {l:.4f} epochs {e} "
                  f"agg_mistakes {n_optimal_mistakes(policy, agg_items)} "
                  f"[{time.time()-t1:.0f}s]", flush=True)
            pending = None
            checkpoint(it + 1)
    except KeyboardInterrupt:
        where = pending if pending is not None else it
        print(f"\ninterrupted in round {where}; resume with "
              f"--resume {ckpt_dir or '<ckpt-out>'}", flush=True)
        raise

    if not stop_reason:
        stop_reason = "max_iters"
    if stop_reason == "max_iters":
        checkpoint(max(it + 1, start_round), stop_reason=stop_reason)
    else:
        checkpoint(it, done=True, stop_reason=stop_reason)
    repair_s = prior_repair_s + (time.time() - t_repair)
    final_cert = certify_report(policy, oracle)
    print(f"repair: W {final_cert['worst_draw_rate']:.0f} "
          f"B {final_cert['best_draw_rate']:.0f} "
          f"D {final_cert['random_draw_rate_exact']:.4f} "
          f"added {len(agg)} [{repair_s:.0f}s]", flush=True)

    ckpt_path = ""
    if ckpt_dir is not None:
        f = ckpt_dir / "c4_gru.pt"
        _atomic_save(policy.state_dict(), f)
        ckpt_path = str(f)
        print(f"saved {f}", flush=True)

    # ---- matched random controls ----
    added = []
    prev = 0
    for e in log:
        added.append(e["aggregate"] - prev)
        prev = e["aggregate"]
    counts, running = [], 0
    for a in added:
        running += a
        counts.append(running)
    counts = counts[1:]  # first certification has no additions yet

    pools = {}
    modes = [m.strip() for m in args.control_mode.split(",") if m.strip()]
    if not args.no_controls:
        if "all" in modes:
            raw = sorted(set(g.random_play_histories(4000, seed=2026)))
            pools["all"] = [h for h in raw if _is_constrained(h, oracle)]
        if "tree" in modes:
            raw = sorted(set(g.optimal_play_histories(oracle, limit=60000)))
            pools["tree"] = [h for h in raw if _is_constrained(h, oracle)]
    if not args.no_dagger:
        for dm in [m.strip() for m in args.dagger_mode.split(",") if m.strip()]:
            if dm not in modes:
                modes.append(dm)

    controls = []
    if not args.no_controls:
        for mode in modes:
            for cseed in [int(s) for s in args.control_seeds.split(",") if s.strip()]:
                torch.manual_seed(1000 + cseed)
                c_policy = GRUPolicy(args.hidden)
                c_policy.load_state_dict(policy_state0)
                rng = random.Random(cseed)
                pool = pools.get(mode)
                c_agg, c_set = [], set()
                c_log = [{
                    "round": 0,
                    "worst_draw_rate": base_cert["worst_draw_rate"],
                    "best_draw_rate": base_cert["best_draw_rate"],
                    "random_draw_rate_exact": base_cert["random_draw_rate_exact"],
                    "n_states": base_cert["n_states"],
                    "n_frontier": base_cert["n_frontier"],
                    "n_violations": base_cert["n_violations"],
                    "added": 0,
                }]
                t0c = time.time()
                for r, target in enumerate(counts):
                    if mode in ("dagger", "dagger-full"):
                        visited = dagger_histories(c_policy, oracle, rng,
                                                   args.dagger_games)
                        cands = [h for h in dict.fromkeys(visited)
                                 if tuple(h) not in c_set]
                        need = (len(cands) if mode == "dagger-full"
                                else max(0, target - len(c_agg)))
                        if need < len(cands):
                            rng.shuffle(cands)
                            cands = cands[:need]
                    else:
                        cands = sample_pool(pool, target - len(c_agg), rng)
                    for h in cands:
                        if tuple(h) not in c_set and not g.terminal(g.board_of(h)):
                            c_agg.append(tuple(h))
                            c_set.add(tuple(h))
                    c_items = items_from(c_agg, oracle)
                    stop = (replay + c_items if args.stop == "both"
                            else c_items if args.stop == "agg" else replay)
                    l, e = fit(c_policy, replay + c_items * args.agg_repeat,
                               args.control_epochs or args.epochs, args.lr,
                               args.seed + 100 + r + cseed, stop_items=stop)
                    cert = certify_report(c_policy, oracle)
                    c_log.append({
                        "round": r + 1,
                        "worst_draw_rate": cert["worst_draw_rate"],
                        "best_draw_rate": cert["best_draw_rate"],
                        "random_draw_rate_exact": cert["random_draw_rate_exact"],
                        "n_states": cert["n_states"],
                        "n_frontier": cert["n_frontier"],
                        "n_violations": cert["n_violations"],
                        "added": len(c_agg),
                        "epochs": e,
                        "visited": len(cands) if mode in ("dagger", "dagger-full") else None,
                        "replay_mistakes": n_optimal_mistakes(c_policy, replay),
                    })
                    print(f"  control {mode} s{cseed} round {r + 1}/{len(counts)}: "
                          f"W {cert['worst_draw_rate']:.0f} "
                          f"D {cert['random_draw_rate_exact']:.3f} "
                          f"epochs {e} [{time.time() - t0c:.0f}s]", flush=True)
                controls.append({
                    "pool": mode,
                    "seed": cseed,
                    "log": c_log,
                    "final": c_log[-1] if c_log else {},
                    "elapsed_s": time.time() - t0c,
                })
                print(f"control {mode} s{cseed}: W {c_log[-1]['worst_draw_rate']:.0f} "
                      f"B {c_log[-1]['best_draw_rate']:.0f} "
                      f"D {c_log[-1]['random_draw_rate_exact']:.4f} "
                      f"added {c_log[-1]['added']}", flush=True)

    out = {
        "args": vars(args),
        "environment": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "when_utc": datetime.now(timezone.utc).isoformat(),
        },
        "game": {"rows": g.ROWS, "cols": g.COLS, "win": g.K,
                 "boards": board_count},
        "replay_histories": len(replay_hist),
        "replay_items": len(replay),
        "twin_teachers": {"draw_rates": twin,
                          "mean": sum(twin) / len(twin),
                          "median": statistics.median(twin)},
        "base": {"certificate": base_cert, "fit_loss": loss, "fit_epochs": eps,
                 "source": base_source},
        "repair": {"log": log, "final": final_cert, "aggregate": agg,
                   "elapsed_s": repair_s, "checkpoint": ckpt_path,
                   "stop_reason": stop_reason, "resumed_from": resume_dir},
        "controls": controls,
        "elapsed_s": prior_elapsed + (time.time() - t_start),
    }
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
