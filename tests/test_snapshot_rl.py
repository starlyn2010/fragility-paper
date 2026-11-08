"""Contract tests for runs/snapshot_rl.py, the archiver that protects the RL stage.

The repair arm writes its JSON on the final round only, so every round before that exists
as printed text in runs/_rl_c4x4.log and nothing else. A shutdown therefore loses the
rounds unless the snapshotter rebuilt them, and the snapshotter is only worth running
before a machine is switched off if these four promises hold:

1. every `[repair it N]` line in the log becomes a recovered round, with the numbers the
   line actually printed --- a dropped or mistyped round is silent data loss;
2. the recovered rounds are contiguous from 0 and the last one is the log's final round,
   so a gap in the middle cannot be mistaken for a complete history;
3. the archive holds the reconstructed table plus each artifact it claims, and is a
   readable zip;
4. the snapshot never writes where the live process writes, because overwriting
   data/c4_rl_repair_x4_s0.json mid-run would corrupt the result it is meant to protect.
"""
import importlib.util
import json
import pathlib
import re
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("snapshot_rl", REPO / "runs" / "snapshot_rl.py")

# runs/ is pruned from the sdist, so an installed artifact has neither this script nor the
# log it reads; the promises only hold in a checkout that has run, or is running, the RL arms.
pytestmark = pytest.mark.skipif(
    not (REPO / "runs" / "_rl_c4x4.log").exists(),
    reason="no RL run to snapshot in this checkout")


def _module():
    mod = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(mod)
    return mod


def test_every_repair_line_in_the_log_becomes_a_recovered_round():
    mod = _module()
    text = mod.LOG.read_text(encoding="utf-8", errors="replace")
    printed = re.findall(r"\[repair it\s+(\d+)\].*", text)
    data = mod.parse(text)
    assert len(data["repair_rounds"]) == len(printed), "a round was dropped"
    by_round = {r["round"]: r for r in data["repair_rounds"]}
    for line in text.splitlines():
        if not line.startswith("[repair it"):
            continue
        n = int(re.search(r"it\s+(\d+)", line).group(1))
        want = dict(D=float(re.search(r" D ([\d.]+)", line).group(1)),
                    frontier=int(re.search(r"frontier\s+(\d+)", line).group(1)),
                    violations=int(re.search(r"viol\s+(\d+)", line).group(1)),
                    aggregate=int(re.search(r"agg\s+(\d+)", line).group(1)),
                    mistakes=int(re.search(r"mistakes\s+(\d+)", line).group(1)))
        for k, v in want.items():
            assert by_round[n][k] == v, "round %d read %s wrong" % (n, k)


def test_the_recovered_rounds_are_contiguous_and_end_at_the_log_last_round():
    mod = _module()
    text = mod.LOG.read_text(encoding="utf-8", errors="replace")
    rounds = mod.parse(text)["repair_rounds"]
    assert [r["round"] for r in rounds] == list(range(len(rounds)))
    assert rounds[-1]["round"] == int(re.findall(r"\[repair it\s+(\d+)\]", text)[-1])


def test_the_snapshot_is_strict_json_and_the_zip_holds_what_it_claims():
    mod = _module()
    assert mod.main() == 0
    table = json.loads(mod.OUT_DIR.joinpath("rl_stage3_rounds_from_log.json")
                       .read_text(encoding="utf-8"), parse_constant=
                       lambda x: (_ for _ in ()).throw(ValueError("not strict JSON: " + x)))
    assert table["repair_rounds"], "the snapshot recovered no rounds"
    with zipfile.ZipFile(mod.OUT_DIR / "rl_stage3_snapshot.zip") as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        assert "artifacts/rl_stage3_rounds_from_log.json" in names
        for rel in mod.COPY:
            if (REPO / rel).exists():
                assert rel in names, "the archive lost " + rel
        assert len(table["repair_rounds"]) == len(
            re.findall(r"\[repair it\s+\d+\]", mod.LOG.read_text(encoding="utf-8",
                                                                 errors="replace")))


def test_the_snapshot_never_writes_where_the_running_repair_arm_writes():
    mod = _module()
    live = (REPO / "data" / "c4_rl_repair_x4_s0.json").resolve()
    targets = {mod.LOG, mod.STAGE_LOG} | {(REPO / rel).resolve() for rel in mod.COPY}
    assert live not in targets
    assert str(mod.OUT_DIR) not in str(live), "the snapshot dir is the arm's output dir"
    assert not any(str(p).endswith("c4_rl_repair_x4_s0.json") for p in targets)
