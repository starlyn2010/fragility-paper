# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

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
