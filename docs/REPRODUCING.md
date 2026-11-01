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
```

**Cost note.** The certificate is exact over every optimal teacher tie-break below
each root, so its cost grows with the number of empty cells rather than with board
size. On the two-core laptop that produced these files, the ten-cell seed 0 file took
41,192 s, of which only 6,455 s is the repair loop: the two matched controls are 84% of
the wall time because they rarely satisfy the stopping rule and so spend all 200 epochs
in every round. Seed 1, run with `--no-controls`, certified in 2,502 s (14 rounds,
166 histories). Twelve and fourteen empty cells are out of reach on that machine for the
*complete* arm; `colab/c4_endgame_scaling.ipynb` queues the repair arms, the control arms
and the verifications of a depth as separate processes, one per usable core.

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
# three repair arms on the shared suite, no inline controls
for s in 0 1 2; do
  python src/c4_endgame.py --roots 40 --empties 12 --root-value 0 --branch 3 \
    --hidden 128 --emb 16 --epochs 200 --lr 2e-3 --agg-repeat 60 --stop both \
    --max-iters 24 --seed $s --threads 1 --no-controls \
    --roots-file data/c4_endgame_roots_12_40_s0.json \
    --ckpt-out runs/c4_end12_s$s --out data/c4_end12_s$s.json
done
# then six control arms, two per seed, all independent of each other
for s in 0 1 2; do
  for mode in all tree; do
    python src/c4_endgame.py --roots 40 --empties 12 --root-value 0 --branch 3 \
      --hidden 128 --emb 16 --epochs 200 --lr 2e-3 --agg-repeat 60 --stop both \
      --max-iters 24 --seed $s --threads 1 --roots-file data/c4_endgame_roots_12_40_s0.json \
      --controls-from data/c4_end12_s$s.json --control-mode $mode \
      --control-seeds 1 --control-games 3 --out data/c4_end12_s${s}_$mode.json
  done
done
```

A run that reads `--roots-file` no longer rewrites it, so the suite keeps the name of the
seed that sampled it.

Expected: 4x4 base `W=0` (mean 32.8%, 75 violations) → `W=B=1` after 33
rounds and 1,182 histories, zero replay mistakes; repaired certificate 8,877
states. The two matched controls (10-epoch cap per round, 104 and 68 epochs
used against 116 for the repair arm) add 1,181 histories and stay at `W=0`
(mean 48.2% and 32.5%). Endgame base `W=0` (mean 76.6/78.9/79.4%, 43/42/40
violations for seeds 0/1/2) → `W=B=1` after 8/6/7 rounds and 111/102/95
histories, with one/three/one replay mistakes. All matched controls stay at
`W=0`. Both repaired checkpoints are re-certified independently at `W=1`.
The ten-cell suite is the same protocol two plies deeper: seed 0 starts at
mean 68.2% with 59 violations and 18/40 roots drawn, certifies after 17 rounds
and 214 histories with zero replay mistakes, and its two matched controls (182
and 150 histories) end at mean 73.0% and 87.0% with `W=0`. Seed 1 starts at 48
violations and certifies after 14 rounds and 166 histories. Seeds 1-2 use
seed 0's root suite and the identical refit budget, so the per-seed comparison
is matched.

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
  --files "data/c4x4_ctrl_s0.json:4x4 Connect Four,data/c4_end_ctrl_s0.json:7x6 8-empty,data/c4_end10_s0.json:7x6 10-empty" \
  --panel-width 2.5 --out figs_v2/fig8_scaling.pdf
```

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
