# Reproducing the experiments

All commands are POSIX shell (Git Bash on Windows, Linux, macOS) and match the
argparse definitions in `src/`. Training checkpoints are not versioned; every
evaluation writes `sha256_16` so a row can be tied to exact weights.

## 0. Reviewer quickstart

Everything that does not require training can be checked in about two minutes,
and the saved evidence can be re-verified without retraining:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q          # 35 tests, ~2.5 min
make verify                        # re-runs the independent verifiers
```

`make verify` reloads the saved checkpoints and recomputes the Tic-Tac-Toe,
Connect Three, 7x6 endgame and Rock-Paper-Scissors certificates with code that
does not share the repair recursion. It needs the `runs/` checkpoints produced
by `make repair`, `make c4` and `make rps`. The paper is rebuilt with
`make paper` (Tectonic) and every table and figure is generated from the JSON
files in `data/`, which are versioned.

Each experiment block below states its expected values, so a mismatch is
visible without reading the paper.

## 1. Environment

Tested with Python 3.11.9, `torch 2.14.1+cpu`, `numpy 2.4.6` on Windows 10
(x64), CPU only.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# source .venv/bin/activate        # Linux / macOS

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

Sanity checks (no training):

```bash
python -m pytest tests -q      # 35 tests, ~2.5 min
make smoke                     # imports + CLI --help + the same tests
```

`make` is optional; every target is a thin wrapper around the commands below.

## 2. Train a checkpoint

`ablation_ttt.py` trains a config (A–D) for 5,000 games and writes
`<logs-dir>/ablation_<ID>.pth` plus a JSON results file. `--tiebreak` selects
the mentor's tie-break policy: `first` (deterministic) or `random` (TBR).

```bash
# Fixed deterministic teacher
python src/ablation_ttt.py --configs C --games 5000 --seed 42 \
  --tiebreak first  --logs-dir runs/fixed_C_s42 \
  --results-file runs/fixed_C_s42/results.json

# Tie-Break Randomization (TBR)
python src/ablation_ttt.py --configs C --games 5000 --seed 42 \
  --tiebreak random --logs-dir runs/tbr_C_s42 \
  --results-file runs/tbr_C_s42/results.json
```

The paper trains fixed B/C at seeds 42–44 and the TBR/mixed/noise/episode arms
at the seeds listed in `data/tbr_eval_*.json`. The preserved v1 checkpoints are
in `artifacts/v1_reentreno`.

## 3. Evaluate the twin gap (depth-6 and exact depth-9)

`tbr_eval.py` loads `<dir>/ablation_<CONFIG>.pth` and plays 400 games against
`m6_fixed`, three seeded depth-6 twins, `random_uniform` and `weak_minimax`.
It writes Wilson 95% CIs, mean regret, optimal-action rate, tie/non-tie splits,
and per-game arrays.

```bash
python src/tbr_eval.py \
  --runs "C_s42=runs/fixed_C_s42;C_s43=runs/fixed_C_s43" \
  --config C --games 400 --twin-seeds 1234,5678,9012 \
  --seed 20261027 --out data/tbr_eval_C_s42_43.json
```

`exact_twin_eval.py` repeats the protocol with an exact depth-9 teacher at
every teacher turn: `exact_first` is deterministic, `exact_twin_*` samples
uniformly among exactly optimal actions. Same value function, different
tie-break.

```bash
python src/exact_twin_eval.py \
  --runs "C_s42=runs/fixed_C_s42" \
  --config C --games 400 --twin-seeds 1234,5678,9012 \
  --seed 20261027 --out data/exact_twin_C_s42.json
```

The exact evaluator caches optimal sets per board; a 400-game run takes seconds
after the first call. `--games` must match the game count used in the paper if
you want to compare rates directly.

## 4. Twin ensembles (30 held-out teachers)

```bash
python src/twin_ensemble.py --config C --ckpt runs/tbr_C_s43 \
  --seeds 30 --games 100 --out data/twin_ens_tbr_s43.json
```

`--seeds N` is the number of teachers, not a seed range. `--tie-break random`
(independently per decision, the default) or `episode` (one coherent rule per
game). The paper's lottery table uses both.

## 5. State-space and coverage probes

```bash
# Unique boards/histories visited by a certification suite
python src/coverage_probe.py --config C --ckpt runs/fixed_C_s42 \
  --games 400 --out data/coverage_fixed_C_s42_400.json

# History-level canonical probe (180,361 histories)
python src/history_probe.py --config C \
  --runs "C_s42=runs/fixed_C_s42" --out data/history_probe_C.json

# Board-level canonical probe (2,423 boards)
python src/space_probe.py --config C \
  --runs "C_s42=runs/fixed_C_s42" --out data/space_probe_C.json
```

`coverage_probe.py --ckpt` accepts a checkpoint directory (appends
`ablation_<CONFIG>.pth`) or a file path. `--games` is per opponent.

## 6. Coverage distillation and the memoryless control

`distill_oracle.py` trains a GRU by supervised distillation on nested board
subsets with soft exact-oracle targets:

```bash
python src/distill_oracle.py --config C --coverage 25,100,400,1200,2423 \
  --seeds 0,1 --epochs 400 --out data/distill_C.json
```

It saves `runs/distill_<CONFIG>_cov<N>_s<seed>/ablation_<CONFIG>.pth`. Game
outcomes are then evaluated with `tbr_eval.py`:

```bash
python src/tbr_eval.py \
  --runs "C_distill_cov1200_s0=runs/distill_C_cov1200_s0" \
  --config C --games 200 --out data/tbr_eval_distill_1200.json
```

The memoryless MLP (`memoryless.py`) uses the same 2,423 boards and soft
targets but no recurrence; `eval_memoryless.py` runs the held-out teacher
suite:

```bash
python src/memoryless.py --seeds 0,1 --epochs 1500 --out data/memoryless.json
python src/eval_memoryless.py --ckpt runs/memoryless_s0/mlp.pt \
  --games 100 --twins 30 --coherent 30 --out data/eval_memoryless_s0.json
```

## 7. Worst-case certificate over all optimal teachers

`worst_case_eval.py` enumerates every reachable history once, computes the
agent's greedy action at each history, then runs a memoized backward induction
on `(board, history)`. Learner nodes have one successor; teacher nodes branch
over the exact depth-9 optimal set and record the minimum, maximum, and uniform
mean. It also reconstructs the witness line that attains the minimum.

```bash
# All 21 recurrent checkpoints plus both MLPs (~3 min on CPU)
python src/worst_case_eval.py --preset all \
  --mlp "MLP-s0=runs/memoryless_s0/mlp.pt;MLP-s1=runs/memoryless_s1/mlp.pt" \
  --out data/worst_case.json

# A custom subset
python src/worst_case_eval.py \
  --runs "C42=C:runs/fixed_C_s42;C43=C:runs/fixed_C_s43" \
  --out data/worst_case_C.json

# Optional independent check of the exact uniform mean
python src/worst_case_eval.py \
  --runs "B-v1=B:artifacts/v1_reentreno;C-42=C:runs/fixed_C_s42" \
  --mc-games 20000 --mc-seed 2026 --no-witness \
  --out data/worst_case_mc_check.json
```

Presets are `paper` (nine fixed checkpoints), `tbr` (six TBR runs), `extra`
(episode/mixed/noise/distill), and `all`. `--no-witness` skips witness
reconstruction; `--mc-games` adds a sampled draw rate for comparison with the
exact mean.

## 8. Model-size controls

Three checks separate parameter count from the teacher and the state. All use
exact depth-9 targets.

```bash
# A 3.7x wider GRU (600 hidden) on the same 2,423 canonical histories
python src/distill_oracle.py --config C --coverage 2423 --seeds 0,1 \
  --epochs 400 --hidden 600 --max-minutes 30 --out data/distill_C_h600.json

# Exact certificate for a widened checkpoint (@hidden overrides the width)
python src/worst_case_eval.py \
  --runs "C-h600-s0=C@600:runs/distill_C_cov2423_s0_h600" \
  --no-witness --out data/worst_case_h600_s0.json

# Complete history coverage at the original width
python src/distill_history.py --coverage all --epochs 20 --threads 4 \
  --out data/distill_history_C_full.json
python src/distill_history.py --coverage all --epochs 60 --lr 1e-3 --seed 1 \
  --init runs/distill_hist_C_histall_s0/ablation_C.pth --max-minutes 45 \
  --out data/distill_history_C_full_ft.json
python src/worst_case_eval.py \
  --runs "C-hist-full=C:runs/distill_hist_C_histall_s1" \
  --no-witness --out data/worst_case_histfull.json

# History-level regret for any checkpoint on all 180,361 histories
python src/distill_history.py --eval-only --ckpt runs/fixed_C_s42/ablation_C.pth \
  --out data/history_metrics_controls.json
```

`distill_history.py --coverage all` trains on every history with soft targets
uniform over the exact optimal set; `--coverage canonical` reproduces the
2,423-history control. `--eval-only` skips training. Wall-clock for the long
fine-tune can include OS suspend time; `epochs_run` and the metrics are
authoritative.

Expected: both widths fit the 2,423 training states perfectly, but mean history
regret stays at 0.155--0.158 and `W=0`; full-history coverage reaches 0.005
regret yet `W=0`; the same 323k GRU with random training ties (C-TBR-43) and the
70,665-parameter MLP are `W=B=1`.

## 9. Certificate-guided repair

CGR alternates exact certification with fine-tuning on the violating frontier
histories. Each command below writes the base certificate, the repair log, the
aggregate, and six matched random controls (three from all histories, three from
the optimal-play tree), so the comparison is within one file.

```bash
# C-42: RL checkpoint, 7 histories, 4 rounds
python src/repair.py --config C --ckpt runs/fixed_C_s42/ablation_C.pth --seed 0 \
  --control-seeds 1,2,3 --control-mode all,tree --eval-all \
  --ckpt-out runs/repair_C42_s0 --out data/repair_C42_s0.json

# Full-history GRU (W=0 despite 0.005 history regret): 5 histories, 2 rounds
python src/repair.py --config C --ckpt runs/distill_hist_C_histall_s1/ablation_C.pth \
  --control-seeds 1,2,3 --control-mode all,tree --eval-all \
  --ckpt-out runs/repair_histfull_s0 --out data/repair_histfull_s0.json

# Broken 20-epoch GRU (W=B=0): 3 histories, 2 rounds
python src/repair.py --config C --ckpt runs/distill_hist_C_histall_s0/ablation_C.pth \
  --control-seeds 1,2,3 --control-mode all,tree --eval-all \
  --ckpt-out runs/repair_hist20_s0 --out data/repair_hist20_s0.json

# Independent certificate of a repaired checkpoint (different implementation)
python src/worst_case_eval.py --runs "C-42-repaired=C:runs/repair_C42_s0" \
  --out data/worst_case_repair_C42.json

# Repair-trajectory figure (Fig. 5 in the paper)
python src/repair_figure.py --runs "C-42 (GRU 300)=data/repair_C42_s0.json;Full-history GRU=data/repair_histfull_s0.json;Broken GRU (20 ep)=data/repair_hist20_s0.json" \
  --cols 3 --panel-width 2.45 --out figs_v2/fig6_repair.pdf
```

The fit inside each round uses the 2,423-history canonical replay set, the
aggregate oversampled 60 times, Adam at 5e-4, and at most 12 epochs; it stops
early when the aggregate reaches zero regret. Controls reuse the exact counts of
added histories, rounds, optimizer, and epochs of the repair arm, and sample
their histories without looking at violations. Expected: C-42 7/4, full-history
5/2, broken 3/2; all reach `W=B=1`; every control stays at `W=0`. The same run
also executes two DAgger arms on C-42 with the same stopping test: matched
budget (7 states) certifies 0/3 seeds, and the full visited budget certifies 2/3
(20 states in the passing seeds, 72 in the failing one).

## 10. Connect Three (4x3) certificate and repair

The Connect Three oracle enumerates all 2,715 reachable boards; the replay set
is one canonical history for each of the 933 learner-turn boards. The base GRU
(14,443 parameters) reaches zero mistakes on that replay and is still certified
at `W=0` in all three seeds. CGR stops a refit only when both the 933-item
replay and the aggregate are exactly optimal, which preserves board accuracy.

```bash
python src/c4_repair.py --rows 4 --cols 3 --win 3 --hidden 64 --seed 0 \
  --epochs 300 --lr 2e-3 --max-iters 10 --control-seeds 1,2,3 \
  --control-mode all,tree --ckpt-out runs/c4_canon_s0 --out data/c4_canon_s0.json
python src/c4_verify.py --ckpt runs/c4_canon_s0/c4_gru.pt --hidden 64 \
  --games 2000 --out data/c4_verify_s0.json
python src/c4_figure.py --out figs_v2/fig7_c4.pdf
```

Expected per seed: base `W=0`, `B=1`, `D=11.1/22.2/33.3%`; 8-11 added
histories in 2-5 rounds; final `W=B=D=100%`, zero replay mistakes; all 18
controls at `W=0`; independent verifier `W=1` and 100% against deterministic
orders and 2,000 adaptive optimal teachers. Seed 0 also runs six DAgger arms:
matched budget certifies 2/3 seeds at 11 states, full visited budget certifies
2/3 at 27-32. `make c4` runs all three seeds.

### Connect Four (4x4) and 7x6 endgames

```bash
# 4x4 Connect Four: one-teacher tree replay, exact certificate, CGR
python src/c4_repair.py --rows 4 --cols 4 --win 4 --replay teacher --hidden 64 \
  --epochs 120 --lr 2e-3 --agg-repeat 10 --stop agg --max-iters 40 --seed 0 \
  --control-seeds 1 --control-mode all,tree --control-epochs 10 \
  --ckpt-out runs/c4x4_ctrl_s0 --out data/c4x4_ctrl_s0.json
python src/c4_verify.py --ckpt runs/c4x4_ctrl_s0/c4_gru.pt \
  --rows 4 --cols 4 --win 4 --hidden 64 --games 2000 --out data/c4x4_verify_s0.json

# 7x6 endgames with eight empty cells (draw roots only)
python src/c4_endgame.py --roots 40 --empties 8 --root-value 0 --branch 3 \
  --hidden 128 --emb 16 --epochs 200 --agg-repeat 20 --stop agg --max-iters 12 \
  --seed 0 --control-seeds 1,2,3 --control-mode all,tree \
  --ckpt-out runs/c4_end_ctrl_s0 --out data/c4_end_ctrl_s0.json
# seeds 1-2 reuse seed 0's root suite so the comparison is matched
for s in 1 2; do
  python src/c4_endgame.py --roots-file data/c4_endgame_roots_8_40_s0.json \
    --roots 40 --empties 8 --root-value 0 --branch 3 --hidden 128 --emb 16 \
    --epochs 200 --agg-repeat 20 --stop agg --max-iters 12 --seed $s \
    --no-controls --ckpt-out runs/c4_end_s$s --out data/c4_end_s$s.json
done
python src/c4_endgame_verify.py --ckpt runs/c4_end_ctrl_s0/c4_gru.pt \
  --roots data/c4_endgame_roots_8_40_s0.json --hidden 128 --emb 16 \
  --orders 120 --games 40 --out data/c4_end_verify_s0.json

# Two plies deeper: ten empty cells, same board, same 40-root protocol.
# Seed 0 carries both matched controls; seeds 1-2 reuse seed 0's root suite.
python src/c4_endgame.py --roots 40 --empties 10 --root-value 0 --branch 3 \
  --hidden 128 --emb 16 --epochs 200 --lr 2e-3 --agg-repeat 60 --stop both \
  --max-iters 24 --seed 0 --control-seeds 1 --control-mode all,tree \
  --control-games 3 --ckpt-out runs/c4_end10_s0 --out data/c4_end10_s0.json
for s in 1 2; do
  python src/c4_endgame.py --roots-file data/c4_endgame_roots_10_40_s0.json \
    --roots 40 --empties 10 --root-value 0 --branch 3 --hidden 128 --emb 16 \
    --epochs 200 --lr 2e-3 --agg-repeat 60 --stop both --max-iters 24 \
    --seed $s --no-controls \
    --ckpt-out runs/c4_end10_s$s --out data/c4_end10_s$s.json
done
python src/c4_endgame_verify.py --ckpt runs/c4_end10_s0/c4_gru.pt \
  --roots data/c4_endgame_roots_10_40_s0.json --hidden 128 --emb 16 \
  --orders 120 --games 40 --out data/c4_end10_verify_s0.json
for s in 1 2; do
  python src/c4_endgame_verify.py --ckpt runs/c4_end10_s$s/c4_gru.pt \
    --roots data/c4_endgame_roots_10_40_s0.json --hidden 128 --emb 16 \
    --orders 120 --games 40 --out data/c4_end10_verify_s$s.json
done
```

**Cost note.** The certificate is exact over every optimal teacher tie-break below
each root, so its cost grows with the number of empty cells rather than with board
size. On the two-core laptop that produced these files, the ten-cell seed 0 file took
41,192 s, of which only 6,455 s is the repair loop: the two matched controls are 84% of
the wall time because they rarely satisfy the stopping rule and so spend all 200 epochs
in every round. The two replicas, run with `--no-controls`, certified in 2,502 s (14
rounds, 166 histories) and 2,244 s (7 rounds, 166 histories), so an arm that runs alone
costs a fifth of seed 0's file. Twelve and fourteen empty cells are out of reach on that
machine for the *complete* arm; `colab/c4_endgame_scaling.ipynb` queues the repair arms,
the control arms and the verifications of a depth as separate processes, one per usable
core.

**Opening the notebook.** The repository is private, so the fastest route is Colab ->
*File* -> *Open notebook* -> *Upload*, and pick `colab/c4_endgame_scaling.ipynb` from
disk (or keep it in `MyDrive/c4_scaling` and use the *Drive* tab). The
`colab.research.google.com/github/...` URL also works but asks Colab itself to
authorize against your GitHub account first.

**Nothing else is needed, because the notebook carries its own code.** A cell near the
top holds a base64 zip of the fifteen files the queues need -- the four modules
(`src/c4_game.py`, `src/c4_repair.py`, `src/c4_endgame.py`, `src/c4_endgame_verify.py`),
the four published root suites `data/c4_endgame_roots_{8,10,12,14}_40_s0.json`, the four
blind holdouts `data/c4_endgame_roots_{8,10,12,14}_40_holdoutB.json`, the two eighty-position
unions `data/c4_endgame_roots_{8,10}_80_AplusB.json` the scope-buying arms are pointed at and the
archived seed-1 arm `data/c4_end10_s1.json` that the optional controls section re-derives -- about
340 KB, and
the setup cell rebuilds a working repository from it. The suites are found by a glob rather
than a hand-written list, so sampling a new depth extends every bundle on the next
`build_bundle.py --all`, and `--check --all` is what tells you when a document still quotes
the old count. The suites travel because every
published seed read the same file: a runtime that sampled 40 of its own endgames would be
answering a different question. That copy has `src/` and `data/` and
nothing else --- `colab/build_bundle.py` names those files and no others, and only a
clone would find `runs/` there, because the `runs/*.sh` drivers are tracked --- so the queue
cell creates the folder its arm logs go into rather than assuming the checkout the laptop
has. A fresh runtime therefore needs no sign-in, no token and no upload;
`Cell -> Run all` is the whole procedure. `colab/build_bundle.py`
regenerates that cell from the repository and stamps it with the commit and the sha256 of
every file; `python colab/build_bundle.py --check --all` exits non-zero if the embedded copies
no longer match the files on disk, and `tests/test_colab_notebook.py` enforces the same
rule. The bundle is verified, not trusted: the setup cell recomputes the archive hash and
each file's hash, and a mismatch falls through to fetching the repository, so a
hand-edited bundle cannot silently become the code a paper number was computed from.

**Nothing is stored on the runtime either.** `snapshot()` zips the depth's JSONs, root
suites, checkpoints and logs into `/content/c4_end{E}_results.zip` and hands it to the
browser as soon as one of them changes, so the artifacts leave the machine on every pass of
the queue and no Drive mount, session storage or paste step is involved. Log files are
deliberately outside the signature that decides "changed" --- they grow every two minutes
and would push a new download each pass --- and where the browser will not take the file the
cell prints a warning and leaves the zip in `/content`, because the download is the exit
channel and not the experiment. The section-1 resource cell is guarded for the same reason:
its RAM half comes from `/proc/meminfo`, which a local Jupyter on macOS or Windows does not
have, and where RAM cannot be read the queue holds two arms instead of one per core --- an
out-of-memory kill loses the run, while an idle core only costs time.

**The follow-up notebook.** `colab/build_followup_notebooks.py` writes
`colab/c4_ladder_finish.ipynb`; it is generated, so
the shared machinery - bundle cell, `bootstrap()`, the resource cell, `tick()`, the monitor,
the pump - is the copy `tests/test_colab_notebook.py` already pins down, and
`python colab/build_followup_notebooks.py --check` exits non-zero if the file on disk no
longer matches its generator. Neither notebook asks anything of the person running it: open
it, set the runtime to CPU, `Cell -> Run all`.

- the finisher queues the arms the seed ladder did not run - seeds 7-9 at twelve empty cells
  and 3-5 at ten and eight - against the suites and budgets the published arms used. Seeds
  3-6 at twelve did run there and are archived in `data/`, so this notebook does not repeat
  them; the ladder it replaces is kept at `runs/_superseded_c4_seed_ladder.ipynb`.

The seeds 0-2 at fourteen empty cells - the next point on the curve - are no longer a notebook
of their own; they are the training tail of `colab/c4_finish_all.ipynb`, described below, and
the per-arm document they came from is kept at `runs/_superseded_c4_endgame_14_holdout.ipynb`.

Each arm is one repair against suite A plus two verifications of the saved weights: on suite
A, the forty positions it was certified over, and on suite B, a blind forty positions. B is a
holdout because of what it is not - not the same positions, and not the positions the repair
trained on, since a repair only ever sees positions *below* its roots, with strictly fewer
empty cells. The cell that reads them asserts the two suites are disjoint and prints
`posiciones compartidas 0`; a run where they overlap stops instead of certifying the same
forty positions twice. `runs/make_holdout_suites.py` samples them on the laptop and commits
them, and `random_games` is deterministic in its sample seed (suite A from seed 0, suite B
from `500 + empty cells`), so a runtime that had to sample a depth itself - fourteen is that
depth until the files land - writes the same forty positions, which is the promise
`tests/test_followup_notebooks.py::test_a_runtime_sampling_a_committed_depth_gets_the_same_positions`
checks against the committed data rather than against a description of it.

A new seed is only worth having if it is comparable with the seeds already published at that
depth, so the queue cell carries the budget per depth - 20 aggregated repeats and the `agg`
stopping rule at eight, 60 and `both` at ten and twelve, `--max-iters` 12 and 24 - and the
tests read those numbers out of the archived arm JSONs in `data/`. Nothing is restated from
the paper: if the queue's flags drift from the archived ones at any depth, a test fails.
Fourteen empty cells have no archived arm to copy, so `colab/reads_plan.py` borrows the budget
of the depth below it (`ARM_BUDGET_FROM = {14: 12}`) and
`tests/test_reads_notebooks.py::test_the_fourteen_empty_budget_is_the_twelve_empty_published_one`
checks the fourteen-empty command flag by flag against what twelve ran with.

The artifacts leave the runtime as browser downloads - no Drive mount, no session storage -
with names that cannot collide when both remaining notebooks run in one browser:
`c4_finish_end{E}_results.zip` from the ladder, one per depth, and
`c4_finishall_<gate>_results.zip` from the everything-still-owed document, one per gate that
still has jobs - four of them (`reads_union`, `reads_closure`, `reads_pairing`, `end14`) when the
document was written and, as it stands, only `end14`, because a gate the plan no longer queues is
not a gate the notebook declares. Each zip
carries that gate's repair and verification JSONs, both suites, the checkpoints and the logs.
Both hold every download until a whole gate is on disk - in the ladder by the `PUSH_AT_END`
flag (the test that keeps it honest mutates the flag and the `_done` predicate and requires the
snapshot to go empty), in the everything-still-owed document by a `snapshot()` that only opens a
gate whose planned outputs all exist - because packaging cost the arm wall time in the run where
the download window opened mid-training. Either way
the `ship()` cell at the bottom of the notebook is the emergency exit - run it before closing
the tab if Colab warns that the session is ending, and it will deliver whatever exists, arms
half-done included.

**One notebook that finishes the rest.** A checkpoint that is not in the repository cannot be
re-certified by a reviewer, and `.gitignore` keeps `*.pt` and `runs/**` out, so a document that owes
a read of a weight has to carry that weight. `colab/build_reads_notebooks.py` writes that document,
`colab/c4_finish_all.ipynb`. It opened with forty-eight jobs in one queue - the 16 reads of
the three union arms (the `sec:union` side), the 5 reads that close the last gaps of `tab:blind`
(the suite-A side of eight-cell seeds 1 and 2, and the blind side of twelve-cell seeds 0-2), the
18 reads of the pre-repair checkpoint of nine arms on both suites, and last the three fourteen-empty
arms, each a repair followed by its two verifications - and every one of those thirty-nine reads is
done, along with the eighteen that joined the pairing afterwards on this machine: six twelve-cell
base reads whose checkpoints were already on this disk, and twelve reads of the six arms that
trained before `c4_gru_base.pt` existed, whose bases `runs/_refit_bases.py` re-fitted here from their
own archived flags and accepted only once each re-fit reproduced the certificate the arm logged. So
the document as it stands queues nine, and they are all training:
the 3 fourteen-empty arms, each a repair against the published suite followed by its own read on that
suite and on the blind one. Nothing needs to travel inside, which is why its weights section prints a
sentence saying so instead of an empty table - but when a queue does need weights that already exist,
each one rides inside as base64 pinned by the
`sha256` and byte count the cell itself re-derives before it writes a file: a notebook whose
base64 does not hash to what it announces stops, rather than reading a stranger's weights. The
three per-group read notebooks and the per-arm fourteen-empty notebook this one replaces are gone
from `colab/`; the holdout is kept at `runs/_superseded_c4_endgame_14_holdout.ipynb`.

What a notebook queues is not typed by hand. `colab/reads_plan.py` computes the owed reads and the
untrained arms from
the JSONs already in `data/` - a read is one checkpoint, one suite, one output name - and the
generator, `runs/_reads_audit.py` and `tests/test_reads_notebooks.py` all read that one module,
so a stale queue fails a test instead of spending hours of free CPU re-reading a file that was
archived months ago. Two things make a read worth queueing at all: it has to reproduce the
certificate the arm logged beside it, memo-free, and the union of two disjoint suites has to stay
the minimum of its halves. Both are asserted in the results cell, which prints `COINCIDE` and
`identidad W_U = min(W_A, W_B): OK` and exits non-zero on either refusal, and a repair is checked
against the JSON the runtime itself wrote before its two reads are trusted. Nothing is handed to the
browser until a gate is complete, and `python colab/build_reads_notebooks.py --check` exits
non-zero if a file on disk no longer matches its generator.

Those two promises are the notebook's, and a download is not proof they held. `python
runs/_integrate_reads.py` re-runs them locally before it folds a gate zip into `data/`: a read that
does not reproduce the certificate its arm logged is refused, a union that has stopped being the
minimum of its halves is refused, a suite inside the zip whose sha256 differs from the committed one
is refused per zip - a good copy in another gate does not launder a bad deck in this one, because
the deck is what the read is about - and a file the queue does not owe is refused as a zip from
another generation. It never overwrites an archived JSON, and `--check` prints the same table and
writes nothing. After a successful ingest the queue is stale, so regenerate the notebook before
launching it again. `tests/test_integrate_reads.py` drives all of this against fabricated zips in a
temp directory, with one honest run and a mutant per refusal.

The twelve reads that complete the pairing were made on this machine rather than queued: those arms
ran before `c4_gru_base.pt` was saved, so their base had to be re-fitted, and `runs/_refit_bases.py`
accepts a re-fit only when its certificate is the number the arm logged before its first refit. The
equivalence across runtimes is demonstrated per base rather than assumed, and a re-fit that misses it
is refused before any read of it exists.

**The same queue, on this machine.** Nothing in the derivation of a job is Colab-specific, so
`python runs/_local_queue.py` runs the notebook's queue here: one process per job, the commands the
pump cell would build, flag for flag, and the same two promises checked before a result is kept.
`--check` prints every command and runs nothing. The queue is re-derived from `data/` on every pass,
so an arm that lands frees its slot for its own two reads, and a session that dies mid-arm resumes at
what is still owed instead of re-training a repair that already landed. Transcripts go to
`runs/_localq/<output stem>.log`, and `status.json` and `status.txt` beside them are rewritten from
those transcripts while the processes run - reading a log does not touch the process writing it, so
the snapshot is evidence even if the machine is shut down a minute later. Two guards are measured
rather than assumed: free RAM is read with `GlobalMemoryStatusEx`, which the notebook's helper cannot
ask for on Windows and so falls back to two jobs, and no job starts below 2.5 GB free; `--par`, or the
file `runs/_localq/par`, changes how many run at once without orphaning the ones already training.
`tests/test_local_queue.py` holds the driver to the notebook: every read it would run is the command
the notebook would run, and every repair is the published budget flag for flag, with a mutant per
promise.

If the bundle is ever refused, the setup cell still has the two fetch routes it used to
depend on:

1. *Device flow.* The cell prints `! First copy your one-time code: XXXX-XXXX` and the
   URL; type the code at `github.com/login/device` from any browser where you are signed
   in to the account that owns the repository. A device code does not live long, and
   `failed to authenticate via web browser: context deadline exceeded` means `gh` stopped
   polling before anything was approved -- re-run the cell for a fresh one. The line
   `Failed to copy one-time code to clipboard / No clipboard utilities available` is not a
   failure: a Colab runtime has no clipboard, and the code is printed for you to copy by
   hand.
2. *Archive, no credentials.* `git archive -o fragility-paper.zip HEAD` at the repo root
   (about 1.3 MB: code plus `data/`, no checkpoints), then Colab's file sidebar ->
   *Upload to session*, then re-run the setup cell. Both archive layouts work -- the flat
   one that `git archive` writes and the single-folder one that GitHub's *Download ZIP*
   produces -- and an archive that does not contain `src/c4_endgame.py` is refused rather
   than half-trusted.

Either way the first two cells then print `this notebook carries its own code: commit
XXXXXXX, 6 files, ...` and `repo at /content/fragility-paper | on sys.path: True`, which
together are the proof that a usable copy really landed in the runtime.

`python -m pytest tests/test_colab_notebook.py -q` checks this cell without a runtime: it
parses every code cell, runs the device-flow helper against a `gh` faked to emit the
seventeen escape forms a terminal program can produce, unpacks both archive layouts in a
temp directory, and boots a whole fake runtime -- bundle cell plus `bootstrap()`, with
every `subprocess` call rigged to raise -- from the bundle alone in a temporary directory.
Into that booted copy it then runs the queue cell's own `tick()`, with the launches
recorded rather than spawned, and checks that `runs/` exists afterwards and the first arm
was handed `cwd` inside the copy. It also runs the sampling cells' import pattern in a
child process against the real modules, because the notebook and `src/c4_endgame.py` have
to agree on one module name: importing the game as `src.c4_game` loads a second copy, and
the configured one never reaches the sampler.

The strongest of those checks runs the notebook as a document. A bare runtime booted from
the bundle has every code cell executed in the order the cells appear, which is what
`Cell -> Run all` does, with `subprocess.Popen` recorded instead of spawned (the machine
running the suite is running the researcher's own training job) and a clock that steps an
hour per reading. That is where the arm dependencies get pinned: while no repair JSON exists
only the three repair arms are eligible, so a queue that launched a control or a
verification first would be running it against the wrong starting policy. The same run
asserts the log folder appears, that the hands-free pump returns by itself rather than
holding the runtime for its five hours, that the results table prints from real archived
arms, and that the artifact zip lands in `/content`. Only the section-2 probe is skipped,
because what it exists to do is spend twelve- and fourteen-empty-cell oracle time; its
import pattern is the child-process test above.
The failure modes are asserted to be caught, not just covered: a no-op stripper leaves
`[0m` junk in the captured output, the previous single-folder-only lookup returns nothing
for a `git archive` zip, a bundle whose bytes were changed is refused, a `bootstrap()`
that no longer consults the bundle is found reaching for `gh auth status`, a cell that
imports through the package name raises the same `IndexError` the runtime raised, the queue
cell with its `mkdir` removed dies on the first log it opens, the results cell reading a
`summary` key the verify script never writes prints `None` where the numbers belong, the
unguarded `/proc` read raises on a machine that has no `/proc`, and a pump whose time guard
never fires is stopped by the test rather than left to run.

**Splitting a deep arm across cores.** `--controls-from data/<arm>.json` runs a single
matched control as its own process, against the per-round history budgets recorded in an
archived repair arm and from the same starting weights; `--base-fit` re-derives those
weights for arms archived before checkpoints were saved, and refuses to continue unless
the recovered policy reproduces the archived base certificate. The split is exact:
`tests/test_c4_endgame.py::test_controls_can_run_from_a_saved_repair_and_match_inline`
asserts that the split control log equals the inline one round by round, including after
a `--base-fit` recovery. The ten-cell seed 0 file above carries both controls inline; a
deeper arm writes each arm to its own JSON:

```bash
# seven repair arms on the shared suite, no inline controls (seeds 0-2 on the
# laptop's successor queue, seeds 3-6 through the Colab ladder, one core each)
for s in 0 1 2 3 4 5 6; do
  python src/c4_endgame.py --roots 40 --empties 12 --root-value 0 --branch 3 \
    --hidden 128 --emb 16 --epochs 200 --lr 2e-3 --agg-repeat 60 --stop both \
    --max-iters 24 --seed $s --threads 1 --no-controls \
    --roots-file data/c4_endgame_roots_12_40_s0.json \
    --ckpt-out runs/c4_end12_s$s --out data/c4_end12_s$s.json
done
# then the two control arms of the reference seed, independent of each other;
# the eight- and ten-cell rows carry controls for one seed per depth, and each
# extra seed costs about what its repair arm cost
for mode in all tree; do
  python src/c4_endgame.py --roots 40 --empties 12 --root-value 0 --branch 3 \
    --hidden 128 --emb 16 --epochs 200 --lr 2e-3 --agg-repeat 60 --stop both \
    --max-iters 24 --seed 0 --threads 1 --roots-file data/c4_endgame_roots_12_40_s0.json \
    --controls-from data/c4_end12_s0.json --control-mode $mode \
    --control-seeds 1 --control-games 3 --out data/c4_end12_s0_$mode.json
done
python src/c4_endgame_verify.py --ckpt runs/c4_end12_s0/c4_gru.pt \
  --roots data/c4_endgame_roots_12_40_s0.json --hidden 128 --emb 16 \
  --orders 120 --games 40 --out data/c4_end12_verify_s0.json
```

**Reading one checkpoint twice** is the whole of the blind-suite experiment
(`paper/paper_v2.tex`, `sec:blind` and `tab:blind`), and it needs no training at
all - the verifier takes the saved weights and any suite, so the two suites are read
with the same command and the same flags:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do for suite in s0 holdoutB; do
  python src/c4_endgame_verify.py --ckpt runs/c4_end12_s$s/c4_gru.pt \
    --roots data/c4_endgame_roots_12_40_$suite.json --hidden 128 --emb 16 \
    --orders 120 --games 40 \
    --out data/c4_end12_$( [ $suite = s0 ] && echo verify || echo holdoutB )_s$s.json
done; done
# the paired base control: the same two suites, the weights from before the repair,
# written under the suite's short name (onA / onB). These two lines run again at depths
# 8 and 10 for all six seeds, which is where the other twelve pairs come from.
for s in 0 1 2 3 4 5 6 7 8 9; do
  python src/c4_endgame_verify.py --ckpt runs/c4_end12_s$s/c4_gru_base.pt \
    --roots data/c4_endgame_roots_12_40_s0.json --hidden 128 --emb 16 \
    --orders 120 --games 40 --out data/c4_end12_base_s${s}_onA.json
  python src/c4_endgame_verify.py --ckpt runs/c4_end12_s$s/c4_gru_base.pt \
    --roots data/c4_endgame_roots_12_40_holdoutB.json --hidden 128 --emb 16 \
    --orders 120 --games 40 --out data/c4_end12_base_s${s}_onB.json
done
```

Six of those twenty-two pairs have no `c4_gru_base.pt` to point at, because they trained before
that file was ever written. `runs/_refit_bases.py` re-derives them:

```bash
python runs/_refit_bases.py --check   # print every command it would run, run nothing
python runs/_refit_bases.py           # re-fit six bases, read them on both suites, file them
```

Each command is built from the `args` block of the arm's own run JSON, so the flags cannot drift
from the archived run, and it ends in `--no-controls --base-fit --controls-from data/<arm>.json`:
`--base-fit` stops the arm as soon as the base fit is done and saves `c4_gru_base.pt`, and
`--controls-from` makes the same script recompute the base certificate and compare it with the
archive's, exiting non-zero on any difference. Only then does the driver run the two memo-free
reads above, and it deletes a read that `runs/_integrate_reads.py` rejects.

The eight- and ten-cell blind reads are the same two lines with those depths and seeds
0-5; the depth-8 seed-0 arm reads `runs/c4_end_ctrl_s0/c4_gru.pt`, which is the directory its
repaired checkpoint was saved to before `--ckpt-out` existed, and its file is
`c4_end_verify_s0.json`. Those commands now cover the whole ladder: the batch archived on
2026-11-05 added the two missing suite-A reads at eight cells and the three missing blind
reads of twelve-cell seeds 0-2, so `tab:blind` has twenty-two arms and no cell taken from
the loop's own bookkeeping. Why the `_base.pt` matters: without it, a fall on suite B
could be "the holdout is harder", and the paired base reads are what measures that away rather
than arguing it - which is now done for all twenty-two arms (`c4_end{8,10}_base_s0..5` and
`c4_end12_base_s0..9`, forty-four files). All twenty-two pairs ran here rather than in a runtime,
because a memo-free read is a recursion and not a training run: the six whose base weights this
disk already held took two processes at a time, one core each, about seventeen minutes, and the six
that had no checkpoint to read took the same order of wall time once `runs/_refit_bases.py` had
rebuilt each base and compared it with the certificate the archive logs - eight-cell seeds 0-2
live under `runs/c4_end_ctrl_s0`, `runs/c4_end_s1` and `runs/c4_end_s2`, the directories those arms
saved to before `--ckpt-out` existed. A re-fit is the one number here that cannot be assumed, so it
is checked twice: the driver exits non-zero if the re-derived base certificate differs from the
archived one, and `colab/reads_plan.py` refuses to call a read of the seed-0 arm comparable unless
the suite it points at is the suite that arm certified, which at seed 0 it is - the sample the
script drew in-process, before `--roots-file` existed, is position for position
`data/c4_endgame_roots_8_40_s0.json`, and `tests/test_reads_notebooks.py` holds that branch of the
plan.
`runs/_blind_table.py` prints `tab:blind` from the twenty-two repaired checkpoints and their
forty-four reads, plus the twenty-two paired base rows and their forty-four; it refuses to print
an arm whose loop report and memo-free read disagree on `W`, `B`, `D̄` or roots drawn, refuses
an arm that reaches `W=1` on the blind suite, and refuses an unrepaired base that certifies on
either suite.

**The exchange: the loop aimed at the blind suite.** `tab:blind` shows one direction,
so the claim could still be "suite B is a bad sample". The other direction is one flag:
hand the repair `--roots-file data/c4_endgame_roots_<E>_40_holdoutB.json` and its base
fit, its aggregate and its certificate are all computed over suite B's positions. Then
read every checkpoint on both suites, repaired and unrepaired. Nothing in `src/` changes.

```bash
# six arms: three seeds at eight and at ten empty cells, each depth at its published budget
python runs/_fromB_driver.py            # arms, then the twelve repaired reads
python runs/_fromB_bases.py             # the twelve base reads, both suites, two at a time
python runs/_scope_table.py             # prints tab:fitb from those twenty-four files
```

`runs/_scope_table.py` re-derives the round count from `repair.log` rather than trusting
it: a row is logged at the top of an iteration and carries the certificate *before* that
iteration's refit, so an arm that broke on a certificate is counted at one fewer row than
it logged and an arm the cap stopped is counted at the cap. The script asserts that rule
against the thirteen arms that were published before it existed, which is what
makes it a check and not a summary.

Expected: all six arms certify on suite B (`W=B=1`, mean 100%, 40/40 roots drawn) from
bases at 87.3/85.7/82.3% (eight cells) and 75.7/70.3/69.7% (ten) with 29/28/31 and
56/60/61 open violations, in 7/4/6 refits and 103/77/66 histories at eight cells and
23/16/19 and 247/223/218 at ten. None certifies on suite A: mean 41.4/40.2/29.9% and
27.0/26.4/35.2% with 12/11/7 and 7/6/8 roots drawn, `W=0` in all six. The same weights
before repair read 35.7/32.1/22.3% and 26.0/25.7/33.5% on A, so the paired gain on the
unseen suite is +5.7, +8.1, +7.6, +1.0, +0.7, +1.7 points - positive in every arm, which
is why the paper calls this a scope failure rather than damage. Repair arms 159-192 s
(eight cells) and 2,881-3,083 s (ten) per arm; the twenty-four reads run two at a time in
about 120 s and 135-150 s.

**Certifying a union of two suites.** If the certificate is a statement about a policy
*and* a set of positions, the way to get a guarantee on two suites is to put both inside
the set, and `certify` takes a minimum over roots, so on two suites sharing no board
`W_{A∪B} = min(W_A, W_B)` - a union certificate at `W=1` is a certificate on the blind
suite by definition, not by transfer. The open question is the price, and the two
commands below keep the budget each depth was published with so the price is comparable
to `tab:scaling76`:

```bash
python runs/make_union_suites.py                 # writes the two A-union-B files
python runs/make_union_suites.py --check         # exits 1 if a union is wrong or reordered
python runs/_union_driver.py --depths 8 --phase arms
python runs/_union_driver.py --depths 8 --phase reads
# and the same two lines with --depths 10, which is an hour per arm on this laptop
```

The reads phase runs each saved checkpoint through the verifier three times - suite A,
suite B, and the union - so the identity above is checked on the weights rather than
assumed. `--ckpt c4_gru_base.pt` gives the same three reads of the pre-repair weights,
which is what separates "the extra forty roots helped the base fit" from "the wider
certificate is what closed the blind suite".

Expected, for the three depth-8 arms already on disk: base certificates over the eighty
roots `W=0`, mean 79.26/78.16/80.56% with 67/83/69 violations, then seed 2 `W=B=1` with
80/80 drawn in 4 refits and 161 histories, seed 1 the same in 10 refits and 211, and seed 0
ending on the cap (12 refits, 191 histories) with one violation and 79/80 drawn - that one
root being index 60 of the union, so a blind one. Splitting `repair.per_root` at A's forty
followed by B's forty gives seeds 1 and 2 `W=1`, `D̄=100%`, 40/40 on both suites and seed 0
`W=1`/100%/40-of-40 on A against `W=0`/99.17%/39-of-40 on B, and the two verifier reads
already run for seed 0 reproduce its half of that exactly. The comparison that carries the
claim is the cost: 66-111 histories in 4-8 refits for forty positions at this depth against
161-211 in 4-10 for eighty, with the cap binding for the first time at eight empty cells.
Seed 0 took 399 s, seed 1 376 s, seed 2 457 s, two arms at a time on one core each; a read
is about 120 s. Seeds 1 and 2 therefore stand on the loop's own per-root recursion until
their six reads and the nine base reads run.

A run that reads `--roots-file` no longer rewrites it, so the suite keeps the name of the
seed that sampled it.

Expected: 4x4 base `W=0` (mean 32.8%, 75 violations) → `W=B=1` after 33
rounds and 1,182 histories, zero replay mistakes; repaired certificate 8,877
states. The two matched controls (10-epoch cap per round, 104 and 68 epochs
used against 116 for the repair arm) add 1,181 histories and stay at `W=0`
(mean 48.2% and 32.5%). Endgame base `W=0` (mean 76.6/78.9/79.4/73.2/79.7/74.5%,
43/42/40/51/31/50 violations for seeds 0-5) → `W=B=1` after 8/6/7/9/5/11 rounds and
111/102/95/130/76/111 histories. All matched controls stay at `W=0`. All six depth-8
arms stop on the aggregate alone, so the replay count is free to move
during the loop and each ended with 1/3/1/5/3/0 suboptimal replay decisions; the ten- and
twelve-cell arms stop on replay plus aggregate and keep that count at zero. The four depth-8
arms that have an independent read - seed 0 and seeds 3-5 - re-certify at `W=B=1` with 40/40
roots drawn by 120 column orders and by 40 adaptive teachers per root.

The ten-cell suite is the same protocol two plies deeper: seed 0 starts at
mean 68.2% with 59 violations and 18/40 roots drawn, certifies after 17 rounds
and 214 histories with zero replay mistakes, and its two matched controls (182
and 150 histories) end at mean 73.0% and 87.0% with `W=0`. Seed 1 starts at mean
72.9% with 48 violations and certifies after 14 rounds and 166 histories (2,502 s
for the file, 2,231 s of it the repair loop); seed 2 starts at mean 72.6% with 49
and certifies after 7 rounds and 166 histories (2,244 s and 1,834 s). Seeds 3-5
arrive from `colab/c4_ladder_finish.ipynb` at means 75.3/70.3/68.9% with
38/51/53 violations and 20/16/18 roots drawn, and certify after 9/13/19 rounds and
177/195/233 histories (repair loop 656/917/865 s on one Colab core, 724/983/919 s
for the whole file) with zero suboptimal replay decisions in every logged round.
All six seeds use
seed 0's root suite and the identical refit budget, so the per-seed comparison
is matched, but the depth cost is not: 7 to 19 rounds at ten cells against 5 to 11 at
eight, and over the six the median price of two more empty cells is 1.8 times the rounds
(7.5 to 13.5) and 1.75 times the added histories (106.5 to 186) for the same procedure
and the same 40 roots. All six repaired checkpoints are re-certified by the
memo-free verifier at `W=B=1`, 40/40 roots drawn by 120 column orders and by 40
adaptive optimal teachers per root.

Twelve empty cells is where the cap starts deciding. The ten arms share one
40-root suite and one budget (`--agg-repeat 60 --stop both --max-iters 24`); their
bases are all `W=0`, `B=1` at mean 53.0-71.5% with 84-139 violations and 7-16 of 40
roots drawn, and seven of the ten end certified: seeds 7, 8 and 9 - the three newest,
from `colab/c4_ladder_finish.ipynb` - inside the cap after 21/22/21 rounds and
358/345/388 histories, seed 0 after 22 rounds and 385
histories, seed 3 after 22 and 391, and seeds 4 and 5 only on the `final`
re-certification the loop takes after its 24th refit (their last logged certificate
still holds one and two violations, so a cap of 23 files them as failures - and under that
cap the ten arms report five certifications rather than seven). Seeds 1,
2 and 6 are cut off with one, two and four violations open at mean 99.9%, 97.5% and
99.1%, and the memo-free verifier reproduces every one of the ten outcomes from the
saved checkpoints - each `c4_end12_verify_s$s.json` equals its arm's own final
certificate, which is the check that the missing certificates belong to the policies.
Each arm costs about one Colab core-hour (repair loop 1,710-3,599 s). That is a
statement about the runtime and not about the depth: the same protocol at fourteen
empty cells runs on the two-core laptop when nothing else is on it, at hours per arm. The two matched controls on seed 0 stay at `W=0` at 59.96%
(constrained pool, 223 histories, 112 violations) and 82.44% (optimal tree, 210, 84).
The count for the paper is `runs/_scaling_table.py`: it reads all twenty-two ladder
arms, re-derives each round count from `repair.log` under the same rule
`runs/_scope_table.py` asserts, and only then prints the rows of `tab:scaling76` - it
reproduces the thirteen arms that were published before the script existed and exits
non-zero if any budget flag of an archived arm differs from the depth it is filed under.

And on the blind suite the same weights do not hold: every one of the twenty-two repaired
arms read on both suites is at `W=0` on suite B - mean 33.2-48.3% at eight cells,
18.3-24.8% at ten, 9.1-20.7% at twelve - while holding 97.5-100% on the suite it was
repaired against, and the depth means fall 42.0 / 20.6 / 14.1% with 83/240, 19/240 and
14/400 roots drawn. That is the expected outcome of `sec:blind`, and the twenty-two
`c4_end*_base_s*_on{A,B}.json` pairs are what stops it from being read as "the
holdout is harder": the unrepaired bases already sit at 11.7-40.8% on B, and the
repair's gain there is -12.3 to +16.4 points (mean +1.0, negative in ten of the twenty-two)
against +20.3 to +46.9 on A (mean +32.1, and to `W=1` in nineteen of the twenty-two). Six of
those twenty-two pairs cost more than a read: their arms saved no base checkpoint, so
`runs/_refit_bases.py` re-fitted the base from the flags in the arm's own JSON, and a re-fit whose
certificate does not reproduce the archived number to twelve decimals never reaches the table.

### Tolerance of the approximate certificate

`src/c4_approx_oracle.py` degrades the Connect Three oracle on purpose and re-runs the
same certificate and repair against the degraded values, so the tolerance in
`proposition (Approximate certificates)` is a measured quantity rather than an assumption.
`--ref integer` keeps the win/draw/loss values (smallest gap `Delta=1`); `--ref graded
--gamma 0.9` replaces a terminal payoff by `+/- gamma^plies`, which preserves every
win/draw/loss sign but shrinks `Delta` to 0.0662. `--mode blend` regresses the oracle
toward the random-play value, which moves `epsilon` continuously.

```bash
for s in 0 1 2; do
  python src/c4_approx_oracle.py --mode blend --ref integer --seed $s --threads 1 \
    --dials 0,0.01,0.02,0.0417,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.05,1.2,1.5 \
    --out data/c4_eps_int_s$s.json
  python src/c4_approx_oracle.py --mode blend --ref graded --gamma 0.9 --seed $s \
    --threads 1 --dials 0,0.3,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0 \
    --out data/c4_eps_grad_s$s.json
done
python src/c4_approx_oracle.py --gaps-only --ref graded --threads 1 \
  --out data/c4_graded_gaps.json          # no training: the Delta versus gamma ladder
python src/eps_figure.py --seeds 0,1,2 --out figs_v2/fig10_eps.pdf
python -m pytest tests/test_approx_oracle.py -q
```

`runs/eps_int_sweep.sh` and `runs/eps_stage2.sh` are the same commands chained as
the two queued single-thread stages that actually produced the files on this
laptop, so neither overlaps the other jobs.

Expected: integer - no misranking and no widening of the teacher class up to
`epsilon=0.50` (`Delta/2`), first misranking at 0.60 with `W=1` still, first failure at
0.80 (`W=0`, worst value -1) with the class still closed, first widening at 1.05
(141/868 states) where the repair price goes from 9-20 histories to 121-169 - and one
seed in three still re-certifies at 1.05, with 139 added histories. Graded -
widening from 2.4 `Delta`, repair 32-85 histories up to 5.5 `Delta` and 105-217 beyond,
and past `epsilon=0.392` a stall: the worst value stays at -0.4305 for every larger dial
in every seed while the nominal bound slides from -9.4 to -12.6. In 30 of the 32
uncertified arms across both scales the loop ends with *zero* violating histories left
on the approximate certificate, which is the failure the proposition predicts and the
reason the files also report the value against the exact reference. The bound is
respected at every dial in every seed, 19-36x (integer) and 22-29x (graded) above the
measured deficit. 462-695 s per integer file, 603-1,899 s per graded file, one core.

### Hidden-type Rock-Paper-Scissors

The oracle is exact: six deterministic teacher types, a uniform posterior over
the types consistent with the observed actions, and a Bayes-adaptive dynamic
program over the 19,234 reachable histories. The certificate normalizes the
policy's exact value between the worst deterministic policy and the Bayes
optimum, so `V^=1` iff the policy is Bayes-optimal.

```bash
python src/rps_repair.py --seed 0 --max-iters 12 --threads 2 \
  --ckpt-out runs/rps_s0 --out data/rps_s0.json      # CGR + controls + DAgger
for s in 1 2; do
  python src/rps_repair.py --seed $s --max-iters 12 --threads 2 \
    --no-controls --no-dagger --ckpt-out runs/rps_s$s --out data/rps_s$s.json
done
for s in 0 1 2; do
  python src/rps_verify.py --data data/rps_s$s.json --ckpt runs/rps_s$s/rps_gru.pt
done
python src/rps_figure.py --seeds 0,1,2 --out figs_v2/fig9_rps.pdf
```

Expected: base `V^=0.698/0.709/0.698` with 19/19/20 violations; CGR reaches
`V^=1.000` in all three seeds with 53/39/47 added histories in 7/5/5 rounds and
zero replay mistakes, ending at the Bayes-optimal per-type vector
`(6,7,6,8,7,8)`. Seed 0 also runs three random control pools (copy / all /
optimal-tree) and two DAgger variants, three seeds each, matched per-round:
COPY-tree 0.709, all-histories 0.748, optimal-tree 1.000 (3/3, but its pool is
only 34 histories and saturates), DAgger-all 0.977 (1/3 certified), DAgger-worst
0.969 (0/3, pool saturates at 43). `make rps` runs the whole sequence, the
verifier, and the figure; `python -m pytest tests/test_rps.py -q` checks the
oracle by brute force at horizon 2.

## 11. Statistics, figures, paper

```bash
python src/analyze_tbr.py --inp "data/tbr_eval_C.json,data/tbr_eval_B.json" \
  --out data/tbr_analysis.json
python src/make_figures.py      # writes paper/figs_v2/*.pdf
python src/c4_figure.py --out figs_v2/fig7_c4.pdf
python src/c4_figure.py \
  --files "data/c4x4_ctrl_s0.json:4x4 Connect Four,data/c4_end_ctrl_s0.json:7x6 8-empty,data/c4_end10_s0.json:7x6 10-empty,data/c4_end12_s0.json+data/c4_end12_s0_all.json+data/c4_end12_s0_tree.json:7x6 12-empty" \
  --panel-width 2.3 --out figs_v2/fig8_scaling.pdf
```

A panel path may join files with `+`: the first is the arm and the rest only
supply their `controls` entries. The twelve-empty-cell controls need it because
they were run as separate processes with `--controls-from`, which is the only
way a deep arm's controls fit on the machine at all.

The paper source is `paper/paper_v2.tex` (IEEEtran). The packaged build uses
Tectonic at `.tools/tectonic.exe` (Windows) or `.tools/tectonic`:

```bash
make paper
# equivalent:
.tools/tectonic.exe paper/paper_v2.tex
```

Tectonic writes `paper/paper_v2.pdf`. If `make` finds no local binary it prints
the same command. With TeX Live:

```bash
cd paper && pdflatex -interaction=nonstopmode paper_v2.tex
```

## 12. Provenance and limitations

`docs/DATA_CARD.md` lists every `data/*.json`, its generator, and the exact
caveats: deterministic action selection, effective sample sizes of 5–13 unique
histories, three seeds per condition (no p<0.05), and the corrected legacy
“random” baseline. The v2 evidence is final; the v1 artifacts are kept for
audit and are not used as v2 claims.
