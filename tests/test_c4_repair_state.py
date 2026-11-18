"""A local repair arm that is killed must resume into the run it would have been.

`src/c4_repair.py` spends hours per refit at fourteen empty cells, so an arm that can only be
restarted from zero is an arm the machine cannot finish. `--ckpt-out` now rewrites `state.pt`
after every certificate and every refit, and `--resume` restores the weights, aggregate, log and
round index. What a crash would otherwise hide is tested here: a run resumed once is the
uninterrupted run (same log, same aggregate, same final certificate, same weights), a run killed
during a refit resumes without redoing that certificate, and a resume that ignores the saved round
index does not reproduce either.

The board and model are small enough that the three sessions take seconds; the state format is
the production one, written by the same subprocess command the paper's arms use.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "src" / "c4_repair.py"
ARGS = ["--rows", "4", "--cols", "3", "--win", "3", "--hidden", "16",
        "--epochs", "3", "--agg-repeat", "2", "--threads", "1", "--no-controls"]


def run(argv, script=SCRIPT):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    done = subprocess.run([sys.executable, str(script)] + ARGS + argv,
                          cwd=str(REPO), capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return done


def state_of(directory):
    return torch.load(pathlib.Path(directory) / "state.pt", map_location="cpu",
                      weights_only=False)


def weights_of(path):
    return torch.load(path, map_location="cpu")


def assert_same_arm(full, resumed):
    for key in ("log", "final", "aggregate"):
        assert full["repair"][key] == resumed["repair"][key], key
    assert full["base"] == resumed["base"]
    assert full["twin_teachers"] == resumed["twin_teachers"]
    a = weights_of(full["repair"]["checkpoint"])
    b = weights_of(resumed["repair"]["checkpoint"])
    assert all(torch.equal(a[k], b[k]) for k in a), "the resumed arm is a different policy"


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    """A three-round arm, its two-round prefix, and the prefix's state file."""
    root = tmp_path_factory.mktemp("c4state")
    full = root / "full"
    run(["--max-iters", "3", "--ckpt-out", str(full), "--out", str(full / "out.json")])
    part = root / "part"
    run(["--max-iters", "2", "--ckpt-out", str(part), "--out", str(part / "out.json")])
    return {
        "root": root,
        "full": full,
        "part": part,
        "full_json": json.loads((full / "out.json").read_text()),
        "part_state": state_of(part),
    }


def test_a_run_resumed_between_rounds_is_the_run_it_would_have_been(arms):
    state = arms["part_state"]
    assert (state["next_round"], state["pending_round"], state["done"]) == (2, None, False)
    out = arms["root"] / "resumed.json"
    run(["--max-iters", "3", "--resume", str(arms["part"]), "--out", str(out)])
    resumed = json.loads(out.read_text())
    assert resumed["repair"]["resumed_from"] == str(arms["part"])
    assert_same_arm(arms["full_json"], resumed)


def test_a_refit_killed_after_its_certificate_resumes_without_redoing_it(arms):
    """The pending checkpoint holds the certificate's work but the weights before the refit."""
    full_state = state_of(arms["full"])
    pending = dict(arms["part_state"])
    pending.update({"log": full_state["log"], "agg": full_state["agg"],
                    "next_round": 2, "pending_round": 2, "done": False,
                    "stop_reason": ""})
    directory = arms["root"] / "pending"
    directory.mkdir()
    torch.save(pending, directory / "state.pt")
    torch.save(pending["policy_state"], directory / "c4_gru.pt")
    out = arms["root"] / "pending.json"
    run(["--max-iters", "3", "--resume", str(directory), "--out", str(out)])
    resumed = json.loads(out.read_text())
    assert [e["round"] for e in resumed["repair"]["log"]] == [0, 1, 2], "a round was certified twice"
    assert_same_arm(arms["full_json"], resumed)


def test_a_resume_that_ignores_the_saved_round_index_does_not_match(arms, tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    mutant = source.replace('start_round = int(preset["next_round"])',
                            "start_round = 0", 1)
    assert mutant != source, "the mutation did not apply; the test would prove nothing"
    script = tmp_path / "c4_repair_mutant.py"
    script.write_text(mutant, encoding="utf-8")
    out = tmp_path / "mutant.json"
    run(["--max-iters", "3", "--resume", str(arms["part"]), "--out", str(out)], script=script)
    resumed = json.loads(out.read_text())
    assert resumed["repair"]["log"] != arms["full_json"]["repair"]["log"], (
        "resuming from round 2 but starting at round 0 reproduced the uninterrupted arm, "
        "so the saved round index is not doing anything")


def test_an_external_base_is_restored_not_retrained_on_resume(tmp_path):
    """The RL arm starts from `--init`; a resume must keep that base, not distill a new one."""
    import c4_repair

    c4_repair.g.configure(4, 3, 3)
    base = tmp_path / "base.pt"
    torch.save({"state_dict": c4_repair.GRUPolicy(16).state_dict(), "rows": 4,
                "cols": 3, "win": 3, "hidden": 16, "emb": 8}, base)
    full = tmp_path / "full"
    run(["--max-iters", "2", "--init", str(base), "--ckpt-out", str(full),
         "--out", str(full / "out.json")])
    part = tmp_path / "part"
    run(["--max-iters", "1", "--init", str(base), "--ckpt-out", str(part),
         "--out", str(part / "out.json")])
    out = tmp_path / "resumed.json"
    run(["--max-iters", "2", "--init", str(base), "--resume", str(part), "--out", str(out)])
    resumed = json.loads(out.read_text())
    assert resumed["base"]["source"] == str(base)
    assert_same_arm(json.loads((full / "out.json").read_text()), resumed)


def test_a_finished_arm_is_not_resumed(tmp_path):
    directory = tmp_path / "done"
    run(["--max-iters", "2", "--hidden", "8", "--epochs", "2",
         "--ckpt-out", str(directory), "--out", str(tmp_path / "done.json")])
    state = state_of(directory)
    assert state["done"] and state["stop_reason"] == "no new violations"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    done = subprocess.run([sys.executable, str(SCRIPT)] + ARGS
                          + ["--max-iters", "3", "--hidden", "8", "--epochs", "2",
                             "--resume", str(directory), "--out", str(tmp_path / "again.json")],
                          cwd=str(REPO), capture_output=True, text=True, env=env)
    assert done.returncode != 0
    assert "is a finished arm" in done.stderr


def test_resume_refuses_a_different_model(arms):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    done = subprocess.run([sys.executable, str(SCRIPT)] + ARGS
                          + ["--max-iters", "3", "--hidden", "8",
                             "--resume", str(arms["part"]), "--out", str(arms["root"] / "x.json")],
                          cwd=str(REPO), capture_output=True, text=True, env=env)
    assert done.returncode != 0
    assert "does not match this run" in done.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
