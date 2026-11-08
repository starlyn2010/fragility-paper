"""Print the verifier reads the repository still owes, grouped as the notebook queues them,
and the arms it has to train itself.

The list itself lives in `colab/reads_plan.py`, which the notebook generator and the contract
tests import too: a queue hand-written from a remembered list of reads keeps its old plan after
the repository changes and spends hours delivering nothing that was missing. So this script is
only a report - it decides nothing.

    python runs/_reads_audit.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "colab"))
import reads_plan as rp                                            # noqa: E402

ROWS = rp.reads()
TODO = rp.pending()
print("%d reads in all, %d of them not on disk, %d already filed"
      % (len(ROWS), len(TODO), len(ROWS) - len(TODO)))
print()

for group in rp.GROUPS:
    ship = rp.shippable(group, TODO)
    fit = rp.needs_base_fit(group, TODO)
    print("=== %s: %d reads a runtime can carry, %d whose checkpoint was never saved%s"
          % (group, len(ship), len(fit),
             "" if not fit else " (would need --base-fit, whose cross-runtime equivalence "
                                "this repository has not demonstrated)"))
    for r in ship:
        print("   %-22s %-16s on %s  <- %s" % (r["kind"], r["arm"], r["suite"], r["out"]))
    for r in fit:
        print("   %-22s %-16s on %s  <- %s  NO CKPT at %s"
              % (r["kind"], r["arm"], r["suite"], r["out"], r["ckpt"]))
    if ship:
        ck = rp.checkpoints(ship)
        print("   %d checkpoints, %.2f MB before base64"
              % (len(ck), sum((rp.REPO / c).stat().st_size for c in ck) / 2 ** 20))
    print()

ARMS = rp.arm_rows()
print("=== arms a runtime still has to train, in the order the queue runs them ===")
for E, seeds in rp.ARMS:
    b, donor = rp.arm_budget(E), rp.ARM_BUDGET_FROM[E]
    print("%d empties, seeds %s | the budget the %d-empty arms were published with: epochs %d, "
          "agg-repeat %d, stop %s, cap %d rounds"
          % (E, sorted(seeds), donor, b["epochs"], b["agg_repeat"], b["stop"], b["max_iters"]))
    for r in [x for x in ARMS if x["E"] == E]:
        print("   %-22s %-16s on %s  <- %s  %s"
              % (r["kind"], r["arm"], r["suite"], r["out"],
                 "on disk" if rp.exists(r["out"]) else "never run"))
print()

done = {}
for r in ROWS:
    if rp.exists(r["out"]):
        done[r["kind"]] = done.get(r["kind"], 0) + 1
print("reads on disk by kind:")
for k in sorted(done):
    print("  %-24s %d" % (k, done[k]))
