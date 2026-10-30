#!/usr/bin/env python3
"""rps_verify.py — independent verification of a hidden-type RPS run.

Recomputes the exact certificate from the saved checkpoint, checks that the
policy value equals a direct simulation against the six types, checks the
claimed Bayes value, and checks that every aggregate history is optimal.

Usage:
  python src/rps_verify.py --data data/rps_s0.json --ckpt runs/rps_s0/rps_gru.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import rps_game as g  # noqa: E402
from rps_repair import GRUPolicy, certify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    horizon = int(data["game"]["horizon"])
    oracle = g.Oracle(horizon)

    policy = GRUPolicy(int(data["args"]["hidden"]), int(data["args"]["emb"]))
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    handle = lambda h: policy.act(h)  # noqa: E731

    cert = certify(policy, oracle)
    final = data["repair"]["final"]
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))
        print(("OK   " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))

    check("certificate is exact optimal", cert["normalized"] >= 1.0 - 1e-9,
          f"W={cert['normalized']:.12f}")
    check("json normalized matches", abs(cert["normalized"] - final["normalized"]) < 1e-9)
    check("json value matches", abs(cert["value"] - final["value"]) < 1e-9)
    check("json per-type matches", cert["per_type"] == final["per_type"])
    check("json violations matches", cert["n_violations"] == final["n_violations"])

    vals = g.per_type(handle, horizon)
    check("value equals direct simulation", abs(sum(vals) / g.N_TYPES - cert["value"]) < 1e-9,
          f"{sum(vals) / g.N_TYPES:.12f}")
    check("bayes value matches DP", abs(oracle.value(()) - data["game"]["bayes_value"]) < 1e-9)
    check("min value matches DP", abs(oracle.value_min(()) - data["game"]["min_value"]) < 1e-9)

    bad = 0
    for h in data["repair"]["aggregate"]:
        h = tuple(tuple(x) for x in h)
        if policy.act(h) not in oracle.optimal(h):
            bad += 1
    check("every aggregate history optimal", bad == 0, f"{bad} violations")

    base = data["base"]["certificate"]
    check("base was not certified", base["normalized"] < 1.0 - 1e-6,
          f"W={base['normalized']:.4f}")
    check("repair improved the certificate", cert["normalized"] > base["normalized"],
          f"{base['normalized']:.4f} -> {cert['normalized']:.4f}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
