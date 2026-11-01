# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [Unreleased]

### Added
- `colab/c4_endgame_scaling.ipynb` gained a hands-free queue cell: it calls `tick()`
  every two minutes until the depth's arms are all on disk or five hours pass, and
  since `tick()` mirrors each finished JSON and checkpoint to Drive, the cell can be
  re-run after an interrupt or a session cap without repeating any completed work.
  The device-flow sign-in used by that notebook was tested non-interactively on this
  laptop with Enter fed on stdin exactly as the cell does: `gh` printed
  `One-time code (XXXX-XXXX)` and the URL and then waited for the approval, rather
  than failing on the missing terminal. And every flag the notebook passes to
  `src/c4_endgame.py` and
  `src/c4_endgame_verify.py` was checked against those two argparse blocks, including
  the `c4_gru_base.pt` that `--ckpt-out` writes for the control arms to inherit.
- Approximate-certificate tolerance study (`src/c4_approx_oracle.py`,
  `src/eps_figure.py`, `tests/test_approx_oracle.py`,
  `data/c4_eps_int_s0..2.json`, `data/c4_eps_grad_s0..2.json`,
  `data/c4_graded_gaps.json`). The oracle is
  degraded on purpose - either by truncating depth or by blending toward the
  random-play value - and the same certificate and repair run against the
  degraded values, with `epsilon` reported as the measured max error. Two payoff
  scales, because the proposition's threshold is a spacing property of the game:
  integer payoffs give `Delta=1` and critical `epsilon=1/(2H)=0.042`, graded
  `+/-gamma^plies` payoffs give `Delta=0.066` and `0.0028`; `--gaps-only` records
  that ladder, which is non-monotone in `gamma` (2.4e-4 at 0.5, 6.6e-2 at 0.9,
  5.8e-2 at 0.95, 1 at 1.0). Three thresholds
  appear, in order: argmax misranking at `Delta/2`, widening of the certified
  teacher class at `Delta` (141/868 integer states, 267/868 graded), and the
  proof's own `Delta/(2H)` two orders of magnitude below both. Certification
  survives 14x (integer) and 133x (graded) past the threshold the proof needs;
  the additive bound holds at every dial and is 19-36x looser than the measured
  deficit; above the crossing the graded arms stall at the same -0.4305 while
  the bound slides, and 30 of the 32 uncertified arms across both scales end with
  zero violating histories still on the approximate certificate - the slack of the
  proof showing up as a real loss the loop cannot see.
  `paper_v2.tex` now states the proposition with `Delta` instead of 1, and the
  sentence that claimed every experiment has `epsilon=0` points at the study.
- Ten-empty-cell 7x6 endgame arm: a paper subsection, a scaling table, a third
  panel in the scaling figure, `data/c4_end10_s0.json` and its 40-root suite.
  Base `W=0` (mean 68.2%, 59 violations) certifies after 17 rounds and 214
  certificate-selected histories; both matched controls stay at `W=0` with more
  data than the eight-cell controls used.
- `colab/c4_endgame_scaling.ipynb` for the twelve- and fourteen-empty-cell arms,
  starting with a probe that measures per-root oracle time on the runtime before
  any suite is committed. It now queues a whole depth as a dependency graph of
  separate processes - three repair arms, the two matched controls of the
  reference seed, three verifications - one per usable core, sampled from a
  single shared root suite, resumable across sessions and mirroring every JSON
  and checkpoint to Drive. A GPU is explicitly not useful here: the certificate
  is a pure-Python recursion and the policy is a 57k-parameter GRU.
- `--controls-from`, which runs one matched control as its own process against
  the per-round history budgets recorded in an archived repair arm, and
  `--base-fit`, which re-derives that arm's starting policy when the archive
  predates checkpoint saving and refuses to continue unless it reproduces the
  archived base certificate. Repair arms now archive their pre-repair weights as
  `c4_gru_base.pt`. Splitting is covered by
  `test_controls_can_run_from_a_saved_repair_and_match_inline`, which compares
  the split arms' per-round certificate logs with the inline ones.
- `src/c4_approx_oracle.py` and `data/c4_approx_s0.json`: the first empirical
  probe of the approximate-certificate proposition. A depth-limited minimax with
  exact random-playout leaves is swept over eleven depths on Connect Three; the
  induced tolerance is 1.34 down to 0 and never enters the regime the bound needs,
  so horizon truncation cannot test the threshold.
- Ten-empty-cell replica `data/c4_end10_s1.json`: 48 base violations certify after
  14 rounds and 166 histories, `W=B=1`, 2,502 s of one laptop.
- Second ten-empty-cell replica `data/c4_end10_s2.json`: 49 base violations certify
  after 7 rounds and 166 histories in 2,244 s, so the same procedure on the same
  40 roots costs 7, 14 and 17 rounds across seeds. Table VIII now carries all three
  ten-cell rows, and the paper reports the widening spread instead of one arm.
- `data/c4_end10_verify_s0/1/2.json`: all three repaired ten-cell checkpoints are
  re-certified by the memo-free verifier at `W=B=1`, 40/40 roots over 120 column
  orders and 40 adaptive optimal teachers per root.
- `tests/test_c4_endgame.py`: contract tests for the endgame harness — the root
  sampler keeps only positions with the requested exact value, the certificate
  orders worst/uniform/best per root and reports one violation per history, the
  matched control pools hold constrained learner-turn histories, and the CLI
  writes the result keys the paper quotes and reuses a saved `--roots-file`.

### Fixed
- `colab/c4_endgame_scaling.ipynb` failed at its first real cell with
  `ModuleNotFoundError: No module named 'src'`: a Colab kernel keeps `/content` on
  `sys.path` and `%cd` does not change it, so `import src.c4_game` resolved locally but
  not in the runtime. Every cell that imports or launches a job now pins the clone root,
  inserts it into `sys.path`, and passes an explicit `cwd` to `Popen`, with a guard that
  names the cell to run when the clone is missing.
- A run that reads its suite with `--roots-file` no longer rewrites it under its
  own seed's filename, so a shared suite keeps the name of the seed that sampled
  it and deeper replicas cannot silently fork the comparison.
- The constrained-history control at ten empty cells was quoted as 73.1% in the
  first draft of the docs; the certificate value is 73.0%.

## [2.0.1] — 2026-10-30

### Fixed
- Citation metadata now uses the Zenodo concept DOI (`10.5281/zenodo.21077401`,
  all versions) instead of the v1 version DOI.

## [2.0.0] — 2026-10-30

### Added
- Certificate-Guided Repair (CGR) for solved games, with the frontier-sufficiency
  lemma, the soundness-and-progress proposition, and matched random controls
  drawn from the full history space and from the optimal-play tree.
- Connect Three (4x3) and 4x4 Connect Four domains with exact certificates,
  three-seed repairs, independent verifiers and matched controls.
- 7x6 endgame certificate over a fixed 40-root suite, three seeds.
- Hidden-type Rock-Paper-Scissors domain with an exact Bayes-adaptive oracle,
  three CGR seeds, matched random controls and two DAgger baselines.
- DAgger baselines on Tic-Tac-Toe and Connect Three, both with CGR's per-round
  budget and with the full visited-state budget.
- Approximate-certificate proposition: with an eps-accurate oracle, greedy play
  on the eps-frontier loses at most 2*eps*H.
- `.zenodo.json`, Dockerfile, `make verify`, and a coverage-enabled CI job.
- Data card and reproduction guide covering every experiment and its expected
  numbers.

### Changed
- The paper now leads with the constructive method (CGR) instead of the
  diagnostic alone, and reports the audit of the original study as a separate
  section.
- The RPS certificate is stated for the Bayesian objective under a known prior,
  not as a worst case over adaptively chosen opponents.

### Fixed
- Rebuilt every table and figure from versioned JSON evidence; the repository
  tracks all JSON files that appear in the paper.
- Corrected the v1 claims about collapse against stochastic opponents, the ACT
  slowdown, and the Rock-Paper-Scissors cycle (see the audit section).

## [1.0.0] — 2026-06-30

- Initial study of tie-breaking fragility in recurrent policies trained against
  a deterministic minimax teacher.
