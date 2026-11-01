# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [Unreleased]

### Added
- Ten-empty-cell 7x6 endgame arm: a paper subsection, a scaling table, a third
  panel in the scaling figure, `data/c4_end10_s0.json` and its 40-root suite.
  Base `W=0` (mean 68.2%, 59 violations) certifies after 17 rounds and 214
  certificate-selected histories; both matched controls stay at `W=0` with more
  data than the eight-cell controls used.
- `colab/c4_endgame_scaling.ipynb` for the twelve- and fourteen-empty-cell arms,
  starting with a probe that measures per-root oracle time on the runtime before
  any suite is committed.
- `tests/test_c4_endgame.py`: contract tests for the endgame harness — the root
  sampler keeps only positions with the requested exact value, the certificate
  orders worst/uniform/best per root and reports one violation per history, the
  matched control pools hold constrained learner-turn histories, and the CLI
  writes the result keys the paper quotes and reuses a saved `--roots-file`.

### Fixed
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
