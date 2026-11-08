"""Snapshot the RL stage to disk without touching the running process.

The repair arm writes its JSON only on the final round, so a shutdown mid-run loses every
round except the printed lines. This reads runs/_rl_c4x4.log, rebuilds the round table from
those lines, and archives it together with the artifacts that already exist. Safe to run at
any moment: it opens the log read-only, never writes where the process writes, and is
re-runnable.

    python runs/snapshot_rl.py
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
LOG = REPO / "runs" / "_rl_c4x4.log"
STAGE_LOG = REPO / "runs" / "_rl_stage3.log"
OUT_DIR = REPO / "artifacts"
ROUND_RE = re.compile(
    r"\[repair it\s+(?P<round>\d+)\] W (?P<W>[\d.]+) B (?P<B>[\d.]+) "
    r"D (?P<D>[\d.]+) frontier\s+(?P<frontier>\d+) viol\s+(?P<violations>\d+) "
    r"agg\s+(?P<aggregate>\d+) mistakes\s+(?P<mistakes>\d+)")
FIT_RE = re.compile(
    r"fit loss (?P<loss>[\d.]+|nan) epochs (?P<epochs>\d+) "
    r"agg_mistakes (?P<agg_mistakes>\d+) \[(?P<seconds>\d+)s\]")
RL_BLOCK_RE = re.compile(
    r"\[\s*(?P<games>\d+)\] reward (?P<reward>-?[\d.]+) draw (?P<draw>[\d.]+)% "
    r"entropy (?P<entropy>[\d.]+) W (?P<W>[\d.]+) B (?P<B>[\d.]+) "
    r"Dbar (?P<Dbar>[\d.]+)% viol (?P<violations>\d+) "
    r"boardmistakes (?P<board_mistakes>\d+) \[(?P<seconds>\d+)s\]")
BASE_CERT_RE = re.compile(
    r"base W (?P<W>\d+) B (?P<B>\d+) D (?P<D>[\d.]+) states (?P<states>\d+) "
    r"frontier (?P<frontier>\d+) viol (?P<violations>\d+)")

# files the run has already finished writing; missing ones are skipped, not fatal
COPY = ["data/rl_c4x4_s0.json", "data/rl_c4x4_s0.pt",
        "runs/_rl_c4x4.log", "runs/_rl_stage3.log"]


def parse(text: str) -> dict:
    lines = text.splitlines()
    rounds = []
    for m in ROUND_RE.finditer(text):
        d = m.groupdict()
        after = lines[lines.index(m.group(0)) + 1:]
        fit = FIT_RE.match(after[0].strip()) if after else None
        if fit:
            d.update(fit.groupdict())
        # the round in progress has no fit line yet
        for k in ("round", "frontier", "violations", "aggregate", "mistakes",
                  "epochs", "seconds"):
            if k in d:
                d[k] = int(d[k])
        for k in ("W", "B", "D", "loss", "agg_mistakes"):
            if k in d:
                d[k] = float(d[k])
        rounds.append(d)
    rl_blocks = []
    for m in RL_BLOCK_RE.finditer(text):
        d = m.groupdict()
        for k in ("games", "violations", "board_mistakes", "seconds"):
            d[k] = int(d[k])
        for k in ("reward", "draw", "entropy", "W", "B", "Dbar"):
            d[k] = float(d[k])
        rl_blocks.append(d)
    base_cert = next((m.groupdict() for m in BASE_CERT_RE.finditer(text)), None)
    if base_cert:
        for k in ("W", "B", "states", "frontier", "violations"):
            base_cert[k] = int(base_cert[k])
        base_cert["D"] = float(base_cert["D"])
    return {"rl_blocks": rl_blocks,
            "base_cert": base_cert,
            "repair_rounds": rounds,
            "stopping": [l for l in lines if "no new violating" in l]}


def main() -> int:
    if not LOG.exists():
        print("%s does not exist" % LOG)
        return 1
    text = LOG.read_text(encoding="utf-8", errors="replace")
    data = parse(text)
    last = data["repair_rounds"][-1] if data["repair_rounds"] else None
    data["snapshot_of_last_round"] = last
    data["log_lines"] = len(text.splitlines())

    OUT_DIR.mkdir(exist_ok=True)
    table = OUT_DIR / "rl_stage3_rounds_from_log.json"
    table.write_text(json.dumps(data, indent=2), encoding="utf-8")

    zip_path = OUT_DIR / "rl_stage3_snapshot.zip"
    tmp = zip_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(table, table.relative_to(REPO).as_posix())
        for rel in COPY:
            p = REPO / rel
            if p.exists():
                zf.write(p, rel)
    shutil.move(str(tmp), str(zip_path))

    print("rounds recovered: %d%s" % (len(data["repair_rounds"]),
          "" if last is None else "  (last: it %d, W %.0f, D %.4f, viol %d)"
          % (last["round"], last["W"], last["D"], last["violations"])))
    print("wrote %s (%.1f KB) and %s (%.1f KB)"
          % (table.relative_to(REPO), table.stat().st_size / 1024,
             zip_path.relative_to(REPO), zip_path.stat().st_size / 1024))
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        print("zip members: %d, integrity %s"
              % (len(zf.namelist()), "ok" if bad is None else "CORRUPT " + bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
