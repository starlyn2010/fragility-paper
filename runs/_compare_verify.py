#!/usr/bin/env python3
"""Diff a fresh `make verify` run against the archived reads, field by field.

`make verify` used to write its output over `data/*.json`, the files the paper quotes. A rerun on
another machine - or with another torch - could therefore replace the evidence it was supposed to
check, and nothing in the suite would notice. It now writes to `runs/verify/` and calls this
script, which fails unless every certificate field is the archived one.

Only the fields the paper prints are compared. Timings, `args`, `environment` and checkpoint paths
are allowed to differ, because they describe the re-run rather than the certificate. For the
Rock-Paper-Scissors verifier, which writes no JSON, the fresh log must contain no `FAIL` line and
must report every check passing.

    python runs/_compare_verify.py [--fresh runs/verify]
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRESH = ROOT / "runs" / "verify"
ARCHIVE = ROOT / "data"

ENDGAME = ("roots", "worst_draw_rate", "best_draw_rate", "random_draw_rate_exact",
           "n_roots_drawn_worst", "deterministic_orders", "order_mean", "order_min",
           "order_max", "adaptive_games_per_root", "adaptive_mean", "adaptive_min")
C4 = ("exhaustive_worst_case", "order_draw_rates", "order_mean",
      "adaptive_teacher_draw_rate", "adaptive_games")
C42 = ("worst_draw_rate", "best_draw_rate", "random_draw_rate_exact",
       "random_loss_rate_exact", "n_states", "sha256_16")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def diff(label, got, want, keys, failures):
    for key in keys:
        a, b = got.get(key), want.get(key)
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9
        else:
            same = a == b
        if not same:
            failures.append("%s: %s is %r, the archive says %r" % (label, key, a, b))


def read_pair(fresh, name, failures):
    fresh, archive = fresh / name, ARCHIVE / name
    if not fresh.is_file():
        failures.append("%s: missing, so `make verify` did not produce this read" % fresh)
        return None, None
    if not archive.is_file():
        failures.append("%s: the archive this read is checked against is missing" % archive)
        return None, None
    return load(fresh), load(archive)


def check_endgame(fresh, failures):
    got, want = read_pair(fresh, "c4_end_verify_s0.json", failures)
    if got is not None:
        diff("endgame", got, want, ENDGAME, failures)


def check_c4(fresh, failures):
    got, want = read_pair(fresh, "c4_verify_s0.json", failures)
    if got is not None:
        diff("connect-three", got, want, C4, failures)


def check_c42(fresh, failures):
    got, want = read_pair(fresh, "worst_case_repair_C42.json", failures)
    if got is None:
        return
    label = "C-42-repaired"
    g, w = got.get("results", {}).get(label), want.get("results", {}).get(label)
    if g is None or w is None:
        failures.append("C-42-repaired: the label is absent from one of the two files")
        return
    diff("C-42-repaired", g, w, C42, failures)


def check_rps_log(fresh, failures):
    log = fresh / "rps_s0.log"
    if not log.is_file():
        failures.append("%s: missing, so `make verify` did not run the RPS verifier" % log)
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    failed = [line for line in text.splitlines() if line.startswith("FAIL")]
    if failed:
        failures.extend("rps: " + line for line in failed)
    match = re.search(r"(\d+)/(\d+) checks passed", text)
    if match is None or match.group(1) != match.group(2):
        failures.append("rps: the verifier did not report every check passing:\n%s" % text)


def main(fresh: pathlib.Path) -> int:
    failures = []
    check_endgame(fresh, failures)
    check_c4(fresh, failures)
    check_c42(fresh, failures)
    check_rps_log(fresh, failures)
    if failures:
        print("the fresh reads do not match the archive:")
        for line in failures:
            print("  FAIL " + line)
        return 1
    print("every fresh read matches the archive on the fields the paper prints")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default=str(FRESH), help="directory the verifiers wrote to")
    sys.exit(main(pathlib.Path(ap.parse_args().fresh)))
