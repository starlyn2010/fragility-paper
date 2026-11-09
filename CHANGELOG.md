# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [Unreleased]

### Changed
- The paired base control of `tab:blind` is **whole**: twenty-two arms, not sixteen, because the six
  arms it could not reach are now in it. Those six - seeds 0-2 at eight and at ten empty cells -
  trained before `c4_gru_base.pt` was ever written, so pairing them needed a base re-derived, and the
  question was whether this two-core laptop can afford a training run. It can, and the cost model that
  said otherwise was wrong in kind: the **759 s** measured here for one memo-free read at fourteen empty
  cells is the *verifier* - 120 deterministic orders and 40 adaptive games per root - and not the loop's
  own certificate, which costs about 150-260 s at that depth. With the machine free, all three
  fourteen-empty arms were launched here on 2026-11-06 at one thread each. Their base fits reproduce the
  two certificates salvaged from the Colab run that cut out - same loss to four decimals and same epoch
  count (0.5616 in 32, 0.5544 in 28), mean 36.47% and 33.92%, 266 and 384 violating histories, 5 and 4 of
  40 roots drawn - and the first refit - the expensive round, the only
  one that adds hundreds of histories at once - took 989 s at seed 0, 2,207 s at seed 1 and 1,905 s at
  seed 2. That first round is not the price of an arm, and extrapolating from it was wrong. The refit
  returns early only when the policy is exactly optimal on the replay *and* the aggregate (`--stop both`),
  so its cost grows with the aggregate: early rounds bought that in 4-15 epochs (700-4,100 s), but by
  round 12-18 the same refit was still running after 20-22 hours awake, near 190 of its 200-epoch cap at
  about 400 s per epoch with three arms sharing two physical cores. The loop had not converged either -
  worst-case draw rate still 0 in all three, violating histories oscillating rather than falling (seed 0
  189 -> 61, seed 1 32 -> 67, seed 2 164 -> 64). The remaining 11, 6 and 12 rounds of the 24-iteration
  budget are therefore days to weeks, not hours. Whether fourteen empty cells is out of reach for the
  method is still open, because the arms had not finished when this was written; what is settled is that
  it is not cheap. `colab/c4_finish_all.ipynb` stays as the fallback for the same three. Two measurement
  notes: the transcripts carry no timestamps, and 25.23 h of the 52.94 h wall since launch was the machine
  asleep (keeper heartbeat gaps), so those per-round figures are awake time, not `elapsed_s`, which
  includes sleep. The
  six eight- and ten-cell bases, by contrast, were trained on this machine under this interpreter, so
  re-fitting them here is reproducing, not approximating - and that is exactly why the driver accepts
  one only on evidence: `runs/_refit_bases.py` rebuilds each base from the flags in that arm's own run
  JSON (`--no-controls --base-fit --controls-from data/<arm>.json`), stops if the re-fit's certificate
  is not the archived number, and then hands the rebuilt weights to the ordinary verifier on both
  suites and deletes the read if `runs/_integrate_reads.py` rejects it. Twelve reads, twelve decimals,
  about an hour of wall time, nothing sent to a runtime.
- What the wider control says is different in kind, not just in count. The repair's movement on the
  blind suite is now **-12.3 to +16.4 points, a mean of +1.0, negative in ten of twenty-two** against
  +20.3 to +46.9 on the suite it was pointed at (mean **+32.1**, and to `W=1` in **nineteen** of the
  twenty-two), and the depth gradient that was +9.8 / +0.0 / -2.8 over three depths' worth of spare
  arms is **+8.0 at eight cells, +0.4 at ten, -2.8 at twelve** - what travels off-certificate shrinks
  toward nothing as depth grows. `tab:blind` is now eighty-eight memo-free reads (twenty-two repaired
  arms and twenty-two paired bases, each on both suites) and `runs/_blind_table.py` derives the paired
  list from `data/` instead of naming it, so the sixteen rows it printed before this batch were the
  ceiling of an assumption about which arms had saved weights, not a list of arms. In the paper the
  same arithmetic replaced the sentence "the pairing is as complete as the weights allow": `sec:blind`,
  `tab:blind` and the limitations now say twenty-two, and the reproducibility note names the driver
  that re-fits. `colab/c4_finish_all.ipynb` was regenerated against a plan that owes **zero reads**:
  nine jobs, all of them fourteen-empty training, no weights inside. Paper: `tab:blind`, `sec:blind`,
  `sec:limitations`.

### Added
- `runs/_local_queue.py` - the notebook's queue, run on this machine instead of a runtime. Nothing in
  `colab/reads_plan.py`'s derivation of an arm is Colab-specific: the budget is read off the archived
  twelve-cell arms, the suites are committed files, and the commands are the ones the pump cell
  builds, flag for flag. The driver re-derives `arm_rows()` on every pass, so a session that dies
  mid-arm resumes at the jobs still owed; it writes a transcript per job under `runs/_localq/` and
  rewrites `status.json` and `status.txt` from those transcripts while the processes run, which is a
  snapshot that never touches the process producing it. Each read is checked with the ingestion
  guard's own `check_reads` before it is left in `data/`, and deleted if it disagrees. `--check`
  prints the commands and runs nothing. Two measured guards replace the notebook's guesswork about
  this machine: free RAM is read with `GlobalMemoryStatusEx` rather than assumed unavailable on
  Windows, and no job starts below 2.5 GB free; `--par`, or the file `runs/_localq/par`, changes how
  many run at once without orphaning the ones already training.
- `tests/test_reproducibility.py` - the promise that a clean clone can run what the prose names,
  asked of git rather than of the author's disk. Fifteen scripts under `runs/` are cited across
  `README.md`, `CHANGELOG.md`, `docs/` and the paper; the test asserts each exists and that
  `git check-ignore` does not hide it, and that `MANIFEST.in` ships the drivers while still pruning
  the weights. It also caught its own naive reading of the paper: long `\texttt` paths are broken
  with `\allowbreak`, which is typography and not part of the filename.
- `tests/test_local_queue.py` - the two promises that driver makes and nothing else checked. Its
  command lines are compared against the ones the notebook's own `verify()` and `repair()` build
  from the same plan rows, and a repair is also compared flag by flag against the archived `args` of
  a published arm at that depth - the anchor that does not drain, since `reads()` comes from
  `LADDER_ARMS` and still has rows once every output is on disk. `owed()` is asked three times about
  the same paired base read with three different disks, and has to answer differently each time:
  weights missing, weights present, result already filed. Both comparisons were run against a
  mutated driver (`--orders 119`, and `owed()` without its checkpoint test) and each mutant turned
  exactly its own test red. Writing it also surfaced two facts the budget table does not carry:
  ten-cell seed 0 ran at `--threads 3` and with its controls inline, so neither flag is part of
  `ARM_FLAGS`, and the test asserts the driver's own choice rather than the archive's.
- `runs/_refit_bases.py` - the driver described above, with `--check` to print every command it would
  run, a transcript per subprocess under `runs/_basefit/` so a killed session still leaves evidence,
  and the two refusal rules (a re-fit that does not reproduce the archived base certificate, a read the
  ingestion guard rejects) exercised on all six arms in this batch.
- `tests/test_metadata.py` - seven assertions that the citation files say what the artifacts they point
  at say: the CFF root uses only 1.2.0 keys and a `type` the schema allows, neither of the preprint's
  two DOIs is ever the root DOI and both are attributed to the preprint, the README paragraph carrying
  them names them as the earlier v1 preprint, the README's page count is read out of the committed PDF
  with `pypdf` rather than trusted, `CITATION.cff` and `pyproject.toml` declare the same version, and
  the contact address is spelled the same in all five files that carry it. Thirteen mutations - eight
  of them the bugs exactly as they were found - each turned its own test red, with all seven controls
  green beforehand and the four mutated files restored and sha256-verified afterwards. Writing them
  caught a stale mutation of its own: `README.md` is CRLF, so a search string with a bare newline
  matched nothing and would have been reported as a test that fails to notice; the script now refuses
  a mutation that does not change the file. `pypdf` joins the dev extra and `requirements-dev.txt` for
  the page count.

### Fixed
- The citation metadata named the wrong paper and was invalid. `CITATION.cff` carried
  `doi: 10.5281/zenodo.21077401` at the root, and that DOI resolves to *The Fragility of Optimal-Agent
  Training* - the v1 preprint this version audits and partly retracts - so GitHub's "cite this
  repository", and anyone following it, would cite the superseded result as version 2.0.1. The file
  was also rejected outright by the CFF 1.2.0 schema: `type: article` is not one of
  `software`/`dataset`, and `year` is not a root property at all, two errors that left the widget with
  nothing to render. The DOI now lives in a `references` entry that names the preprint and states that
  this manuscript has no DOI of its own yet, and the root says `type: software` with
  `date-released: 2026-10-30`. Checked against the official schema: two errors before, none after.
  Three more claims disagreed with their artifacts and now agree - `pyproject.toml` said 2.0.0 while
  the tree is tagged v2.0.1, the README advertised a 29-page PDF beside a 31-page one, and its header
  labelled the preprint DOI "all versions". The reference is pinned to the preprint's **version** DOI
  `10.5281/zenodo.21077402`, which cannot drift if a later deposit is filed under the concept, and the
  concept DOI `10.5281/zenodo.21077401` is recorded beside it as the all-versions identifier; both were
  read off the record itself. The contact address was also spelled two ways - `starlyneliezerrosario`
  in `pyproject.toml`, `starlynelizerrosario` in `CITATION.cff`, `README.md` and both paper sources -
  and is now the former everywhere, confirmed by the author. `paper/paper_v2.pdf` was rebuilt for it:
  31 pages, zero Overfull boxes, and the address extracted from page 1 is the corrected one.
- The notebook's training gate could never be ingested. `runs/_integrate_reads.py` built the list of
  files a zip may carry from `reads_plan.pending()` plus the *repair* rows of `arm_rows()`, but a
  planned arm's two reads live in `arm_rows()` as well - `reads()` only knows depths that already have
  an archived arm - so the `end14` zip's `c4_end14_verify_s0.json` and `c4_end14_holdoutB_s0.json`
  were refused as files "the queue does not owe". Had they been owed, they would have been validated
  against nothing: the expectation came from `data/<arm>.json`, which is inside that same zip and not
  on disk yet. Both halves are fixed together, because either alone leaves a gate shipping an arm and
  two unchecked reads. The arm is checked first - depth, seed, deck and every budget flag against
  `reads_plan.arm_budget`, which reads them off the twelve-cell archives rather than a typed table -
  and then handed to its own reads as their expectation; it lands byte for byte as the loop wrote it
  instead of re-serialised. `--only end14` now works with no read zip in Downloads, because a gate
  that trains is the only work behind it. Seven tests were added to `tests/test_integrate_reads.py`
  (25 in the module) and seven mutations of the guard were run against it: each mutation is noticed,
  including the two that reproduce this bug.
- A new arm's own read was compared against the wrong certificate. `reads_plan.expected()` recognised
  only `repaired-on-` and `union-repaired-` as reads of repaired weights, but `arm_rows()` calls the
  same measurement `arm-read-on-A` - so the guard every local read passes through,
  `runs/_integrate_reads.check_reads`, would have measured a fourteen-empty arm's repaired policy
  against the **base** certificate the loop started from and refused a correct read. The notebook's
  results cell had the rule right in its file-reading branch and wrong in its copied-literal branch
  (`'repaired' in kind`), which is the same disagreement in a second place; both now decide once,
  with the same prefixes. `tests/test_integrate_reads.py` asserts both directions, because either
  alone would pass with the comparison pointed at the wrong tier: the honest read is accepted and a
  read of the base weights is refused. Reverting the one-line fix turns it red, with the guard
  reporting `el loop decia 0.500` against a read of 1.000.
- Every driver the paper cites was invisible to git. `.gitignore` excludes `runs/**` because that is
  where checkpoints, transcripts and scratch folders land, and gave back only `runs/*.sh` - so all
  eighteen `runs/*.py` emitters and guards, twelve of them named in the paper or the docs, were
  ignored, and "run this after a clean clone" was false for every one. `MANIFEST.in` compounded it
  with `prune runs`, which threw the drivers out of the sdist as well. Both now keep the code and
  exclude the artifacts: `!runs/*.py`, and `recursive-exclude runs *` followed by the two includes.
  Nothing is committed by this change; it makes the files committable, which is what was missing.
- `colab/reads_plan.py` raised `KeyError: 'roots_file'` on `data/c4_end_ctrl_s0.json`, the one archive
  whose `args` predate that flag: the earliest arms sampled their certification suite in-process, so
  the plan could not say which suite a read of them is comparable against. It now derives it - at seed
  0 the in-process sample is position for position the published `data/c4_endgame_roots_8_40_s0.json`,
  which is what the re-fit's certificate match confirms and what makes the read of that arm a check
  rather than a guess - and reports any other seed's sample as not comparable instead of pointing a
  read at forty positions the arm never certified. `tests/test_reads_notebooks.py` keeps that branch
  alive by scanning `data/` for archives without the flag, so the test deletes itself when the last of
  them does. Reverting the fix turns it red.

### Changed
- The paired base control of `tab:blind` covers **sixteen** arms rather than thirteen, and the debt
  is now training only. The three twelve-cell seeds that ran before the holdout was cut
  (`c4_end12_s0/s1/s2`) had their `c4_gru_base.pt` on this disk all along, and a memo-free read is a
  recursion, not a training run: six reads, two processes at a time, one core each, seventeen minutes
  at about five minutes a read. `runs/_blind_table.py` emits sixteen paired rows and still asserts
  both of its promises - no unrepaired base certifies anywhere (`W=0` in all thirty-two base reads),
  and the loop's own `base.certificate` reproduces the verifier on suite A in all sixteen. The three
  new bases sit at 58.23/53.01/55.46% on A and 22.47/12.73/19.63% on B, so widening the control
  widened the story it tells: the blind-suite movement after repair is now **-12.3 to +16.4 points,
  a mean of +0.1, negative in eight of sixteen** against +20.3 to +46.9 (mean +34.6) on the suite the
  repair was pointed at, and the depth gradient is steeper than thirteen arms showed it - +9.8 mean
  at eight cells, +0.0 at ten, **-2.8 at twelve**. Where the pairing cannot be closed is said rather
  than skimmed: the six arms at eight and ten cells that ran before `c4_gru_base.pt` was ever saved
  would need a base re-fitted, which is a training run and not a read. `colab/c4_finish_all.ipynb`
  was regenerated against the emptied plan and now queues **nine jobs, all of them the
  fourteen-empty arms**, with no reads and no weights inside; its prose, its gate list and its pump
  text are derived from the plan the queue was built from, so a document with nothing to re-read
  stops promising reads instead of shipping a stale count. Three generator bugs came out of that:
  the packer's comment asserted three read gates draining before training starts in a notebook with
  one gate, the packer printed "hasta 1 zips", and the weights sentence wrote
  "4 trained checkpointcheckpoints". `tests/test_reads_notebooks.py` now checks every read, weight
  and gate promise against a *shape* document as well as the shipped one - with an empty queue those
  tests would have gone quietly true - and the prose test reads both shapes of every sentence, the
  one that quotes a number and the one that says the number is zero. Paper: `tab:blind`, `sec:blind`.

### Added
- The thirty-nine memo-free reads the second `c4_finish_all.ipynb` run brought back are in the
  repository and in the paper, and three of the claims they bear on changed direction rather than
  size. `runs/_integrate_reads.py` is what folds a gate zip into `data/`: it re-runs the notebook's
  own two promises locally before writing a byte - a read that does not reproduce the certificate
  its arm logged stays in the browser, a union that stops being the minimum of its halves is
  rejected, and a zip carrying a suite whose sha256 differs from the committed one refuses the
  whole ingestion because the comparison would no longer be about the same forty positions. It
  also refuses to write over an archived file, so re-running it after a partial ingest is a report
  and not a second measurement. On the paper's side: `tab:blind` reports all **twenty-two** arms,
  zero of twenty-two blind certifications and Wilson upper **0.149**; the paired base control
  covers **thirteen** arms and is now emitted, not hand-typed, because `runs/_blind_table.py`
  prints both blocks of the table and asserts, for each of the thirteen, that the training loop's
  own base certificate and the independent verifier agree on suite A; and the union section
  closes its own confound - all three arms re-read per suite and on the eighty-position whole, and
  all three *base* weights read the same way sit at `W=0` on **both** suites (79.5/74.1/77.0% on A,
  79.0/82.2/84.2% on B), so a doubled sample certifies nothing by itself. What the data weakens is
  said in the same breath: with thirteen paired arms instead of four, the blind-suite movement
  after repair is -10.7 to +16.4 points, negative in five of thirteen, so the paper no longer says
  repair does not damage an unseen suite. It says the A-to-B fall is already in the base weights
  and the refit changes that gap by about nothing. The PDF rebuilt clean, and `README.md`,
  `docs/DATA_CARD.md` and `docs/REPRODUCING.md` carry the same twenty-two arms, the same thirteen
  controls and the same 15 jobs still owed.
- `colab/c4_finish_all.ipynb`, one notebook that finishes everything the repository still owes,
  from [`colab/build_reads_notebooks.py`](colab/build_reads_notebooks.py). It opened with
  forty-eight jobs in a single queue - 16 reads of the three union arms on suite A, on suite B and
  on the eighty-position whole, repaired and unrepaired; 5 reads closing the last gaps of
  `tab:blind` (the memo-free suite-A side of eight-cell seeds 1 and 2, and the blind side of
  twelve-cell seeds 0-2, the three arms that predate the holdout suites); 18 reads of nine more
  arms' `c4_gru_base.pt` on both suites, which took the paired control from four arms to thirteen;
  and at the tail the three fourteen-empty arms, each a repair followed by its two verifications.
  Thirty-nine of those jobs have run and are archived, and because the queue is re-derived from
  `data/` rather than remembered, six jobs it could not have asked for at the start joined it:
  the twelve-cell seeds 0-2 have blind reads now, so their paired base reads became owed. What is
  left is fifteen jobs and three checkpoints - the six `c4_end12_s{0,1,2}/c4_gru_base.pt` reads on
  both suites, then the three fourteen-empty arms. The reads carry no training,
  and the queue trains only what is missing. It carries each checkpoint they read as sha256-pinned
  base64, because `.gitignore` keeps `*.pt` and `runs/**` out of the repository, and the cell
  recomputes each hash and byte count before it writes a file: a notebook whose base64 does not hash
  to what it announces stops rather than reading a stranger's weights. What it queues comes from
  one module, `colab/reads_plan.py`, reading `data/` at build time - the audit
  (`runs/_reads_audit.py`) and `tests/test_reads_notebooks.py` recompute the same list, so a stale
  queue fails a test instead of spending free CPU re-reading an archived file - and the results cell
  refuses twice: a memo-free read that does not reproduce the certificate the arm logged beside it
  is the wrong weights, and a union that stops being the minimum of its two disjoint halves is not
  the union `sec:union` argues about. A repair is checked the same way before its own two reads run,
  against the JSON the runtime wrote. `colab/build_bundle.py` now carries the `_80_AplusB` suites
  beside the `_40_` ones, so a union read never has to sample a suite itself.
- `colab/c4_ladder_finish.ipynb` delivered its nine arms and they are in the paper: seeds 3-5
  at eight empty cells, seeds 3-5 at ten and seeds 7-9 at twelve, each under the budget its
  depth was published with, each archived with a repair JSON, a memo-free read of the saved
  weights on its own suite and a blind read on suite B. The ladder is now 22 arms and the three
  headline counts change shape. At eight cells **six of six certify** in 5-11 rounds for
  76-130 histories (the three new ones in 9/5/11 rounds from bases at 73.2/79.7/74.5%). At ten
  cells **six of six certify** in 7-19 rounds for 166-233, and over the six the median price of
  two more empty cells is 1.8x the eight-cell rounds (7.5 to 13.5) and 1.75x the added histories (106.5
  to 186). At twelve cells **seven of ten certify** (Wilson 0.40-0.89, against 0.25-0.84 at
  seven seeds), the three new seeds closing *inside* the cap at 21/22/21 rounds and
  358/345/388 histories, while the three failures stay capped at one, two and four violations -
  and a cap of 23 would report five of ten, not seven. On the blind side `tab:blind` now holds
  **all twenty-two arms**: zero of twenty-two reach `W=1` on the suite they were not repaired
  against (Wilson upper 0.149), the depth means fall 42.0/20.6/14.1% with 83, 19 and 14 of 240,
  240 and 400 blind roots drawn, and on four of the ten twelve-cell arms - seeds 0, 2, 3 and 7 -
  the blind suite reports `B=0` as well as `W=0`. Every one of the twenty-two has an independent
  memo-free read of its own suite and of suite B, so no cell of the table comes from the training
  loop's own bookkeeping and the `‡` marking is gone; the five reads that closed that last gap -
  the suite-A side of eight-cell seeds 1 and 2 and the blind side of twelve-cell seeds 0, 1 and 2,
  the three that predate the holdout suites - came out of `colab/c4_finish_all.ipynb`, not from
  this notebook.
  Two scripts emit the tables instead of hand-typed numbers: `runs/_scaling_table.py` re-derives
  all 22 rows from the run JSONs, asserts each arm's budget flags against the depth it is filed
  under, and reproduces the thirteen arms published before the script existed;
  `runs/_blind_table.py` emits *both* blocks of `tab:blind` - the twenty-two repaired arms and
  the thirteen paired base reads - refuses to print an arm (repaired or base) whose loop report
  and memo-free read disagree on `W`, `B`, `D̄` or roots drawn, and refuses an unrepaired base
  that certifies. What the nine arms nearly close: thirteen arms now have the paired
  base-on-A/on-B control rather than four, because each of the nine this notebook delivered saved
  a `c4_gru_base.pt` that rode home in the zip, `colab/c4_finish_all.ipynb` read those nine twice,
  and four twelve-cell arms were already paired. Still owed: six reads - twelve-cell seeds 0, 1
  and 2, both suites each - which the same notebook queues, and which are the last thing standing
  between `tab:blind` and a control on every arm. The six arms that no read can pair are the first
  three at eight cells and the first three at ten: they ran before a base checkpoint was ever
  saved, so pairing them would take a re-fit rather than a read.
- The paper has a twelve-empty-cell section (`sec:twelve`) and a fourth scaling panel. The
  depth is where the method first fails, and the reason is a budget rather than a stall: seed
  0 certifies with 385 histories in 22 rounds, while the arms that do not spend all 24 rounds
  the cap
  allows stop with *one*, *two* and *four* violating histories left - 39, 38 and 38 of 40 roots
  drawn in the worst case, means 99.9%, 97.5% and 99.1% - which the independent memo-free
  verifier reproduces from the saved checkpoints, so the miss is the policy and not the
  recursion. The
  section records the shape too: the certifying seed sits between two and seven violations for
  fifteen consecutive rounds before one refit clears them, and it reports the outcome as a
  count with its Wilson interval rather than as a rate. Four seeds landed later under the
  same published budget, so the section counts seven: four certify - seeds 3 and, with no
  round to spare, 4 and 5, whose last logged certificate still held one and two violations
  and which clear on the re-certification taken immediately after the loop - and seeds 1, 2
  and 6 are cut off with one, two and four violations left. 4 of 7 is 0.25-0.84, still too
  wide to separate "this depth usually closes" from "this depth usually fails", and the
  cap-sensitivity is now stated in the text: the same seven arms report 2 of 7 under a cap
  of 23.
  `src/c4_figure.py` now accepts `+`-joined control files, which is how a deep panel gets its
  matched controls at all when those controls had to run as separate processes, and
  `docs/DATA_CARD.md` gains the twelve-empty-cell files of every seed that ran, the root
  suites of every depth and the blind holdouts.
- The paper has a subsection the method does not win: `sec:blind` and `tab:blind`, "What the
  certificate does not claim: a blind suite". Every repaired checkpoint is re-read by the
  memo-free verifier twice, on the forty positions it was certified over (A) and on the
  board-disjoint blind forty (B). All ten arms read on both suites hold A's worst case - nine
  certified, one four violations short - and **none reaches `W=1` on B**: the eight-cell seeds
  fall from 100% to 33.2/44.0/48.3%, the ten-cell ones to 18.3/19.1/24.8%, the four twelve-cell
  arms to 11.2/18.5/14.2/20.7%, and on twelve-empty seed 3 the blind suite reports `B=0` as
  well as `W=0`. "B is just harder" was measured and rejected before the sentence was kept:
  the four twelve-cell *base* checkpoints, paired seed by seed, sit at 57.0-71.5% on A and
  11.7-17.3% on B before any repair, so those same four refits buy +27.6 to +43.1 points where
  the certificate was computed and -6.1, +6.8, -1.8 and +3.8 (mean +0.7) where it was not. The
  disjointness argument is in the text rather than assumed: a repair seeds its initial
  aggregate by enumerating below the roots it is handed, so the forty roots *are* training
  data and a same-depth blind position can only be unseen at board level, which is what
  `runs/make_holdout_suites.py` asserts. `data/c4_end{8,10,12}_holdoutB_s*.json` carry the
  reads, and `runs/_blind_table.py` prints the table from them (it began as
  `runs/_blind_detail.py`, which is now superseded).
- `sec:blind` now has the exchange that closes the last escape in it, `tab:fitb` and six more
  arms. One flag reverses the experiment - `--roots-file data/c4_endgame_roots_<E>_40_holdoutB.json`
  makes the base fit, the aggregate *and* the certificate live on the blind suite - so the
  claim "suite B is just a bad sample" is answered by measurement instead of by argument: all
  six arms (seeds 0-2 at eight and ten empty cells) reach `W=1` on B with 40/40 roots drawn and
  none reaches `W=1` on A, where they read 41.4/40.2/29.9% and 27.0/26.4/35.2%. The
  twenty-four verifier reads also fix the direction of the verb: the same weights before repair
  read 35.7/32.1/22.3% and 26.0/25.7/33.5% on A, so repair *raises* the suite it was never
  aimed at by +0.7 to +8.1 points in every arm. It is a scope failure, not damage, and the text
  says so. The pairing is stated with its limit, because the base fit imitates the positions
  enumerated below its own roots: a B-aimed base starts at 22.3-35.7% on A where the A-aimed
  bases start at 68.2-79.4% on theirs, so the arms pair by seed, depth and budget and not by
  weights. `runs/_fromB_driver.py` runs the arms and their repaired reads, `runs/_fromB_bases.py`
  the base reads, and `runs/_scope_table.py` prints the table - after asserting, against all
  thirteen rows of `tab:scaling76`, that its rule for counting rounds (a log row carries the
  certificate *before* its refit, so an arm that broke on a certificate logs one row more than
  it refitted) is the rule the published table used.
- `runs/make_union_suites.py` and `runs/_union_driver.py` take the next step the scope claim
  invites: put both suites *inside* the certificate. `data/c4_endgame_roots_{8,10}_80_AplusB.json`
  is A followed by B - `build()` refuses to write it if the two suites share a board, and
  `--check` re-validates the count and the order - and a repair handed
  that file certifies against eighty positions at the budget its depth was published with.
  Because `certify` minimises a per-root worst case over roots, `W_{A∪B} = min(W_A, W_B)` on two
  disjoint suites, so a union certificate is a joint guarantee by definition and the only
  scientific question is its price; the driver reads every checkpoint on A, on B and on the
  union, which checks the identity on the weights instead of assuming it, and the same three
  reads of `c4_gru_base.pt` separate the extra forty training positions from the wider
  certificate.
- **The price, measured at eight empty cells.** Three union arms (`data/c4_end8U_s{0,1,2}.json`)
  start from base certificates of 79.26/78.16/80.56% mean draw with 67/83/69 violations over the
  eighty roots, and two of them close: seed 2 reaches `W=B=1` with 80/80 roots drawn in 4 refits
  and 161 added histories, seed 1 in 10 refits and 211. Seed 0 spends the whole published cap -
  12 refits, 191 histories - and its last certificate holds one violation, whose root index (60)
  is inside suite B's block, so the root the eighty-position loop never closed is a blind one.
  Splitting `repair.per_root` at the suite boundary gives seeds 1 and 2 `W=1`, `D̄=100%`, 40/40 on
  *both* suites - the identity, read off the loop's own recursion - and seed 0 `W=1`/100%/40-of-40
  on A against `W=0`/99.17%/39-of-40 on B, which the memo-free verifier re-reads exactly
  (`data/c4_end8U_s0_{A,B}.json`). Against the 76-130 histories in 5-11 refits that certify forty
  positions at this depth, scope costs roughly twice as much and is the first thing at eight empty
  cells to run an arm into the round cap. The sixteen verifier reads behind that sentence - seeds
  1-2 per suite and on the union, the three bases on all three - have since landed, so the crossing
  is reported from the independent recursion and the doubled-sample confound is measured rather than
  argued away; the Limitations entry states the depth, seed count and reading path the remedy was
  measured at.
- Two standalone Colab notebooks for the rest of the scaling curve, generated rather than
  typed. `colab/build_followup_notebooks.py` writes `colab/c4_ladder_finish.ipynb` (the arms
  the seed ladder did not run: seeds 7-9 at twelve empty cells, 3-5 at ten and eight) and
  `colab/c4_endgame_14_holdout.ipynb` (seeds 0-2 at fourteen empty cells - that queue now runs
  inside `colab/c4_finish_all.ipynb`, and the document is archived at
  `runs/_superseded_c4_endgame_14_holdout.ipynb`), each
  arm followed by two verifications of the saved weights: one on the suite it was certified
  over and one on a blind forty-position holdout. They copy the cells
  `tests/test_colab_notebook.py` already pins down - bundle, `bootstrap()`, resource,
  `tick()`, monitor, pump - so one fix reaches all three documents, and
  `python colab/build_followup_notebooks.py --check` exits non-zero if a notebook on disk no
  longer matches its generator. Neither asks anything of the person running it: open, set
  CPU, `Cell -> Run all`. What they hand to the browser differs on purpose:
  `c4_holdout14_end{E}_results.zip` arrives as each arm finishes, while the finisher sets
  `PUSH_AT_END = True` and holds every download until a whole depth is on disk, because zipping
  into the browser while a depth trains takes time away from the training that is still
  running - `c4_finish_end{E}_results.zip` per finished depth, and the `ship()` cell at the
  bottom of the notebook is the emergency exit for a session that is being cut short. The
  superseded ladder is kept at `runs/_superseded_c4_seed_ladder.ipynb`, not deleted: its seeds
  3-6 at twelve cells are the arms `sec:twelve` now reports.
- `runs/make_holdout_suites.py` samples the blind suites on the laptop and commits them -
  forty positions per depth, sample seed `500 + empty cells`, wave by wave into a `.part`
  checkpoint so an interrupted suite resumes - and refuses to write a suite whose positions
  are not forty-empty-cell non-terminal draws, or that shares a board with suite A. Suite B
  is a holdout because a repair only sees positions *below* its roots, which have strictly
  fewer empty cells, so board-level disjointness is the whole of the argument that the blind
  suite was never shown to the learner. `--check` exits non-zero on a committed suite that is
  wrong, and the sampled suites for eight through fourteen empty cells are now in `data/` and
  inside the notebook bundles: `colab/build_bundle.py` carries thirteen files, about 250 KB,
  and `--check --all` checks all three notebooks. The suites are globbed, not listed, so a
  newly sampled depth joins the bundle by itself and `--check` is what notices when a
  document's claimed file list has gone stale.
- The fourteen-cell pair landed on 2 November, so the fourteen-empty queue no longer has
  to sample its own suite: the bundle carries both `c4_endgame_roots_14_40_s0.json` and its
  blind twin, the suites cell reads the committed file and only falls back to sampling if a
  runtime cannot find it. `python runs/make_holdout_suites.py --check` reports all eight
  suites valid and disjoint from their A counterparts. The notebooks' results cell also quotes
  the right interval now: three certifying arms are 0.44-1.00, not 0.21-1.00, which is what
  `wilson(3, 3)` in that cell prints.
- `tests/test_followup_notebooks.py` holds the promises specific to those notebooks (the fourteen
  empty cells now have their own file, `tests/test_reads_notebooks.py`), each
  with a mutant that breaks it the way an accident would: a bare runtime boots the bundle and
  runs a queue end to end; a suite B planted from suite A, a suite of the wrong depth, a
  losing root and a finished position each stop the run; a verification never starts before
  the repair whose checkpoint and JSON it reads, and removing that `needs` guard changes what
  launches; the blind suite is never given to a repair; every planned seed gets both
  verifications; a new seed repeats the budget its depth was published with - eight used 20
  aggregated repeats and the `agg` stopping rule, ten and twelve 60 and `both` - read out of
  the archived arm JSONs, with the mutation of one depth onto another checked flag by flag;
  the two plans together reach ten seeds at twelve and six at ten and eight without repeating
  an archived seed - the archived side is what makes seeds 3-6 count, so a queue that re-ran
  one of them fails; a runtime that samples a depth itself gets the positions the repository
  committed, comparing the sampled boards rather than the description of the seeds; every
  job's output and log appear in the zip, which is how the holdout logs - whose names the
  repair-shaped glob did not match - were found missing; and the download policy is a promise with
  its mutants: the finisher's `snapshot()` returns nothing while
  any planned arm of a depth is missing (flipping `PUSH_AT_END` or stubbing `_done` to `True`
  both fail it), and it still ships one complete zip per depth at the end.
- `runs/snapshot_rl.py` archives the RL stage while it is still running. `src/c4_repair.py`
  writes its JSON on the final round only, so the 30 rounds already computed for the 4x4
  RL base exist nowhere except as printed lines, and one shutdown mid-run would delete them.
  The script reads `runs/_rl_c4x4.log` read-only, rebuilds the per-round table from those
  lines, and writes `artifacts/rl_stage3_rounds_from_log.json` plus
  `artifacts/rl_stage3_snapshot.zip` alongside the artifacts that already exist. It never
  writes where a live arm writes, and re-running it is how the archive keeps up with the run.
  `tests/test_snapshot_rl.py` holds four promises - every `[repair it N]` line becomes a
  recovered round carrying the numbers that line printed, the rounds stay contiguous and end
  at the log's last round, the zip is readable and contains what it claims, and the output
  paths cannot collide with `data/c4_rl_repair_x4_s0.json`.
- `colab/c4_endgame_scaling.ipynb` is standalone: `colab/build_bundle.py` embeds a base64
  zip of the eight files the queue needs - the four modules it runs, the three published
  root suites and the archived arm the controls-from route re-derives, about 200 KB - in a cell of the
  notebook, and the setup cell rebuilds a working copy of the
  repository from it. A fresh Colab runtime therefore needs no GitHub authorisation, no
  token and no uploaded file - open the notebook and run every cell. The bundle carries the
  commit it was built from and the sha256 of each file, and the setup cell recomputes them
  before using anything, so a stale or hand-edited bundle falls through to the fetch routes
  instead of running modified code; `python colab/build_bundle.py --check` exits non-zero
  when the embedded copies no longer match the repository. The clone and device-flow routes
  stay behind it, and the notebook's own markdown now says which step it removed.
- `colab/c4_endgame_scaling.ipynb` gained a hands-free queue cell: it calls `tick()`
  every two minutes until the depth's arms are all on disk or five hours pass, and
  since `tick()` hands each finished JSON and checkpoint to the browser as it lands, the
  cell can be re-run after an interrupt or a session cap without repeating any completed
  work.
  The device-flow sign-in used by that notebook has now run in a real Colab runtime: it
  prints the one-time code and the URL and waits for the approval, and a code that was
  not approved before `gh` gives up polling comes back as
  `failed to authenticate via web browser: context deadline exceeded` followed by the
  notebook's own list of fallbacks - which is a fresh cell run away, not a broken cell.
  And every flag the
  notebook passes to
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
  single shared root suite, resumable across sessions and handing every JSON and
  checkpoint to the browser as it lands. A GPU is explicitly not useful here: the
  certificate is a pure-Python recursion and the policy is a 57k-parameter GRU.
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

### Changed
- `runs/_integrate_reads.py` checks a gate zip's root suites zip by zip instead of once per name,
  which is a real hole closed rather than a refactor: the collection loop merged every zip's copy
  of `data/c4_endgame_roots_*` into one dict and hashed only what survived the merge, so a bad deck
  in the closure gate was overwritten - and its differing sha256 quietly accepted - by the good copy
  the pairing gate happened to carry. The deck is what a read is *about*, so a wrong one makes the
  comparison a different experiment, and the refusal now names the zip it came from.
  `tests/test_integrate_reads.py` is new and covers this file, which had no test at all: seventeen
  promises run against fabricated zips in a temp directory with the plan, the repository and
  `~/Downloads` all sandboxed, and each refusal is checked by its message as well as by a non-zero
  exit - a repaired read that misses the certificate its arm logged, a base read measured against
  the wrong half of the arm, one violation hidden in `n_roots_drawn_worst`, a JSON missing a field,
  a union that is not the minimum of its halves in each of the four ways it can fail to be, a deck
  that is not the committed one, a file the queue does not owe, an archived read, a gate zip that
  never downloaded. The honest zip is run too, on both sides of every mutation, and `--check` and a
  re-ingest are asserted to write nothing, byte for byte.
- The outstanding work is one document, not four. The three per-group read notebooks
  (`c4_reads_union`, `c4_reads_ladder`, `c4_reads_pairing`) and the fourteen-empty holdout queue
  are merged into `colab/c4_finish_all.ipynb`, because a reviewer following the links had to
  judge which of four notebooks still owed what, and the answer moved every time an archive
  landed. The reads keep their order - union, closure, pairing - and the three fourteen-empty arms
  run at the tail of the same queue, so the training starts only after every checkpoint already on
  this disk has been re-read. Downloads are gated by group rather than by arm
  (`c4_finishall_reads_union_results.zip` and its three siblings), which keeps the promise the
  per-arm document broke: nothing is packaged while the CPU is training.
  `tests/test_followup_notebooks.py::test_the_two_colab_documents_do_not_download_over_each_other`
  compares the two remaining notebooks' zip prefixes rather than trusting their names, so a slug
  collision in one browser fails a test. The per-arm notebook is archived at
  `runs/_superseded_c4_endgame_14_holdout.ipynb`, the three read notebooks are deleted, and
  `MANIFEST.in` ships the notebook that exists.
- The scaling notebook carries its outputs to the browser instead of Drive and needs no
  Drive mount anywhere: `snapshot()` zips the depth's JSONs, root suites, checkpoints and
  logs into `/content/c4_end{E}_results.zip` and downloads it when the sizes of the
  non-log artefacts change. Logs stay out of that signature on purpose - they grow every
  two minutes and would push a fresh download on every pass of the queue - and a runtime
  with no browser says so and leaves the zip in `/content` instead of stopping the run.
  `colab/c4_endgame_scaling.ipynb` and `colab/build_bundle.py` are now in `MANIFEST.in`,
  which shipped `src`, `tests`, `docs`, `data` and the paper but not the notebook: the
  artifact a reviewer installs had no route to the deeper arms it points at.

### Fixed
- `colab/c4_endgame_scaling.ipynb` was audited cell by cell for upload, and four defects
  in it were fixed: the results cell asked the verification JSON for a `summary` key that
  `src/c4_endgame_verify.py` never writes, so the published table printed `None` for every
  seed while the real numbers sat under `worst_draw_rate`, `order_min` and `adaptive_min`;
  the resource cell read `/proc/meminfo` with no guard, which is fine on Colab and a
  `FileNotFoundError` in the local Jupyter a reviewer on a Mac or Windows opens the notebook
  in; the header's free-tier paragraph ended mid-sentence ("...hands every finished JSON
  and checkpoint to the"), cut by an earlier edit; and the same header told the reader to
  open the notebook from Drive, which no step of this notebook needs or writes to any more.
  `CHANGELOG.md` and `README.md` still described the outputs as mirrored to Drive and now
  describe the browser route.
- Where the notebook cannot read RAM it holds two arms instead of one per core. The old
  expression was built for a Linux runtime where `MemAvailable` is known, and a machine
  that reports nothing used to crash before reaching that decision; now it survives and
  decides conservatively, because an out-of-memory kill loses the whole run while an idle
  core only costs time.
- `tests/test_colab_notebook.py` gained the check that the notebook actually runs: a new
  test boots a bare runtime from the bundle and executes every code cell in document order,
  the way `Cell -> Run all` does, with `subprocess.Popen` recorded rather than launched (the
  laptop running the suite is running the researcher's own training job on its two cores)
  and the clock stepped an hour per reading. It asserts that only the three repair arms
  launch while no repair JSON exists - the controls and the verifications wait for one, and
  starting them early would run them on the wrong policy - that the queue makes `runs/` and
  opens its logs inside it, that the hands-free pump returns by itself rather than holding
  the runtime, and that the results table prints from real archived arms. The probe cell is
  the one cell skipped, because its job is to spend twelve- and fourteen-empty-cell oracle
  time. Five mutations are asserted to fail it or its neighbours: the `summary` line, the
  unguarded `/proc` read, the missing `mkdir`, a pump whose guard never fires and the
  package-qualified import.
- The scaling notebook's setup cell made Colab show a JavaScript/cookies error instead of
  the GitHub one-time code. `gh` paints its progress with ANSI cursor moves and the cell
  passed that stream straight to the notebook output frame, which cannot render it; the
  stream is now stripped line by line and flushed. `tests/test_colab_notebook.py` runs the
  cell's helper against a `gh` faked to emit the seventeen escape forms a terminal program
  can produce, and a no-op stripper is caught by that same test.
- The archive fallback that the setup cell itself recommends did not work for an archive
  built by `git archive`, because only the code assumed that every zip wraps the repo in
  one folder - `git archive` stores paths flat from the root, so the cell looked for
  `src/c4_endgame.py` inside `.dockerignore` and gave up. Both layouts are now tried, an
  archive without that file is refused with a message instead of a silent `None`, and the
  plain `git clone` runs with `GIT_TERMINAL_PROMPT=0` so a private repo fails with the
  notebook's own guidance rather than `could not read Username`. Both layouts and the
  refusal are covered by `tests/test_colab_notebook.py`.
- The two notebook cells that sample endgames in the kernel died with
  `IndexError: tuple index out of range` inside `c4_game.winner`, at the first root the
  sampler asked the oracle to value. They imported the game as `src.c4_game` while
  `src/c4_endgame.py` imports its sibling by the flat name `c4_game`, so the interpreter
  held two copies of that module: the cell configured its own copy to 6 rows by 7 columns,
  the sampler went on building 4x4 boards from the other copy's `EMPTY`, and a 6x7 solver
  was handed a 16-cell tuple. The cells now import the way every script in `src/` does and
  the setup cell puts `src` on `sys.path` beside the repository root. All three promises
  are tested: no cell may import through the package name (with the mutant import caught),
  the boot really adds `src` to the path, and the flat pattern samples 6x7 roots while the
  package pattern is asserted to raise the same `IndexError`.
- The queue cell died on its first `tick()` with
  `FileNotFoundError: [Errno 2] No such file or directory: 'runs/_colab_c4_end12_s0.log'`.
  The working copy the notebook builds from its own bundle carries `src/` and `data/` and
  nothing else, because `colab/build_bundle.py` ships exactly those two directories, while
  the cell opens each arm's log in `runs/` --- a folder a clone does have, since the six
  `runs/*.sh` drivers are tracked. `src/c4_endgame.py` was never the problem, as
  it creates the parent of every `--out` and `--ckpt-out` it is given. The cell now makes
  the folder before opening the log. `tests/test_colab_notebook.py` boots the bundle into a
  temporary fake runtime, asserts that copy really has no `runs/`, runs the cell's own
  `tick()` with launches recorded instead of spawned, and keeps the promise honest by
  re-running it with the `mkdir` removed to get the original `FileNotFoundError` back.
- `colab/c4_endgame_scaling.ipynb` failed at its first real cell with
  `ModuleNotFoundError: No module named 'src'`: a Colab kernel keeps `/content` on
  `sys.path` and `%cd` does not change it, so `import src.c4_game` resolved locally but
  not in the runtime. Every cell that imports or launches a job now pins the clone root,
  inserts it into `sys.path`, and passes an explicit `cwd` to `Popen`, with a guard that
  names the cell to run when the clone is missing. The root alone turned out not to be
  enough, and the package-qualified form of that import is the bug in the entry above.
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
