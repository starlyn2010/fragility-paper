# Data card — `data/*.json`

Scope: JSON evidence for the tic-tac-toe study *History, Not the Board*
(`paper/paper_v2.tex`). All experiments are synthetic (agents vs. minimax
teachers); there are no human subjects and no personal data. Checkpoints are
`*.pth` files under `runs/`, `checkpoints/`, `artifacts/` and `ablation_logs/`;
they are not versioned. Evaluation files record a truncated SHA-256
(`sha256_16`) so each row can be tied to exact weights.

Most files contain an `args` block with the exact command provenance, an
`environment` block (Python, torch, numpy, platform, threads), and a
`results`/`curvas` payload. The `args` block is authoritative.

## Primary v2 evidence

| File | Produced by | What it records |
|---|---|---|
| `oracle_stats.json` | `src/oracle.py` | Exact depth-9 enumeration of the 2,423 reachable agent-turn boards: `n_tie_boards=1,177` (48.58%), `mean_k=1.9476`, optimal-set-size distribution, and random-vs-fixed tie-break disagreement (62.33% of tie boards, 30.28% of all boards). Teacher-turn statistics (2,097 boards, 937 ties, 28.56% expected twin disagreement) are reported in the paper and recomputed from the same oracle. |
| `history_probe_C_all.json`, `history_probe_C.json` | `src/history_probe.py` | History-level canonical probe. `n_histories=180,361`, `n_boards=2,423`, `n_tie_histories=25,505` (14.14%). Per run: board-regret/optimal and history-regret/optimal, split by tie and non-tie decisions. |
| `space_probe_C.json`, `space_probe_B.json` | `src/space_probe.py` | Board-level canonical probe over all 2,423 agent-turn boards for 14 C runs and 7 B runs. Per run: `mean_regret`, `optimal_action_rate`, tie/non-tie splits. |
| `exact_twin_v1_A/B/C.json` | `src/exact_twin_eval.py` | Exact depth-9 control for the preserved v1 checkpoints: 200 games/opponent, three exact twins (seeds 1234, 5678, 9012). Exact-first teacher and exact twins have the same value function. |
| `exact_twin_B.json`, `exact_twin_C.json` | `src/exact_twin_eval.py` | Same control for the fresh fixed-trained B-42/43/44 and C-42/43/44 checkpoints: 400 games/opponent, three exact twins. C-43 and C-44 are distinct weights but coincide on every evaluated path. |
| `tbr_eval_B.json`, `tbr_eval_C.json`, `tbr_eval_C2.json`, `tbr_eval_C_new.json`, `tbr_eval_B_ep.json`, `tbr_eval_C_ep.json` | `src/tbr_eval.py` | Depth-6 fixed/twin evaluation for every training arm (fixed, TBR random-tie, TBR episode-coherent, mixed, noise): 400 games/opponent, three twin seeds, plus `random_uniform` and `weak_minimax`. Per opponent: W/D/L, Wilson 95% CIs, mean regret, optimal-action rate, tie/non-tie splits, per-game arrays. |
| `twin_ens_fixed_s42.json`, `twin_ens_tbr_s42/43/44.json`, `twin_ens_mixed_s42.json`, `twin_ens_episode_s43.json`, `twin_ens_coherent_tbr_s43.json` | `src/twin_ensemble.py` | 30 held-out twin teachers per checkpoint, 100 games each (3,000 games). Reports mean draw rate and exact range. Used for the seed-lottery table. |
| `coverage_fixed_C_s42.json`, `coverage_fixed_C_s42_400.json`, `coverage_fixed_C_s42_30twins.json`, `coverage_tbr_C_s43.json`, `coverage_tbr_C_s43_400.json`, `coverage_mixed_C_s42_400.json` | `src/coverage_probe.py` | Unique boards and unique histories visited by a certification suite. At 400 games vs M6: 5 boards / 5 histories. At 30 twins × 100 games: 12 boards / 13 histories. Coverage is 0.0028% and 0.0072% of the history space, 0.21% and 0.50% of the board space. |
| `distill_C.json` | `src/distill_oracle.py` | Board-level distillation of a GRU on nested board subsets (25, 100, 400, 1,200, 2,423), two seeds, soft exact-oracle targets. Per model: board regret/optimal rate and full-history metrics. |
| `tbr_eval_distill_1200.json`, `tbr_eval_distill_full.json` | `src/tbr_eval.py` | Game outcomes for the 1,200-board and 2,423-board distilled GRUs: 0/200 vs M6 and twins at 1,200 boards; 100% vs M6 but 32.2% vs twins at full coverage. |
| `history_probe_distill_full.json` | `src/history_probe.py` | Full-history probe of the 2,423-board distilled GRU (`C_distill_full_s0`, `sha256_16=c9b82de9e1eb2ce5`, the same checkpoint as the 32.2% twin draw rate above): 180,361 histories, mean regret 0.1580, optimal-action rate 0.9019, tied-history regret 0.2806 (optimal 0.8566). |
| `memoryless.json` | `src/memoryless.py` | Memoryless MLP (9–256–256–9) on soft exact-oracle targets: board-optimal rate on all 2,423 boards (1.0 for both seeds). |
| `eval_memoryless_s0.json`, `eval_memoryless_s1.json` | `src/eval_memoryless.py` | Held-out teacher evaluation of the MLP: 400 games vs M6, 30 random twins and 30 coherent twins (100 games each), uniform random and weak minimax. |
| `tbr_analysis.json` | `src/analyze_tbr.py` | Group summaries with Wilson CIs and paired sign-flip tests for the TBR arms (C: +0.506, p=0.25; B: −0.142, p=0.75). |
| `worst_case.json` | `src/worst_case_eval.py` | Exact backward-induction certificate over every deterministic depth-9 optimal teacher for 21 recurrent checkpoints and both MLPs. Per run: worst-case draw rate `W`, best-case `B`, exact uniform mean `random_draw_rate_exact`, memoized `(board, history)` state count, and the minimizing witness line. All nine fixed checkpoints: `W=0`, `B=1` (exact mean 25.0% except B-43 37.5%, B-44 50.0%). C-TBR-43 and both MLPs: `W=B=1`. B-TBR-43/44: `W=B=0`. States 27–90 (48 for MLP). |
| `worst_case_mc_check.json` | `src/worst_case_eval.py --mc-games 20000 --mc-seed 2026` | Independent uniform-tie-break simulation for B-v1 and C-42: 25.25% and 25.18% draws vs the exact 25.00%, confirming the backward-induction mean. |
| `distill_C_h600.json` | `src/distill_oracle.py --hidden 600` | Width control: GRU with 600 hidden units (1,185,691 parameters, 3.7×) distilled on the same 2,423 canonical histories with the same deterministic depth-9 targets, two seeds, 400 epochs. Both seeds: board regret 0 and optimal-action rate 1.00 on the training states; mean history regret 0.155/0.157, optimal rate 0.903. |
| `distill_history_C_full.json`, `distill_history_C_full_ft.json` | `src/distill_history.py --coverage all` | Full-history coverage control: the 300-unit GRU trained on all 180,361 histories with soft exact optimal-set targets. Base run: 20 epochs, mean history regret 0.0346. Fine-tune from it: 33 effective epochs at lr 1e-3, mean regret 0.00526, optimal rate 0.99574. The fine-tune's `elapsed_s` includes an OS suspend interval; `epochs_run` and metrics are the authoritative fields. |
| `worst_case_h600_s0.json`, `worst_case_h600_s1.json` | `src/worst_case_eval.py --runs "C-h600=C@600:..."` | Exact certificate for both 600-wide GRUs: `W=0`, `B=1`, exact uniform mean 68.75% (s0) and 50.0% (s1), 90 and 51 memoized states. |
| `worst_case_hist20.json`, `worst_case_histfull.json` | `src/worst_case_eval.py` | Certificates for the full-history GRU before and after the long fine-tune. 20 epochs: `W=B=0` (10 states). Fine-tuned: `W=0`, `B=1`, exact mean 37.5% (42 states), despite 0.005 mean history regret. |
| `history_metrics_controls.json` | `src/distill_history.py --eval-only` | Exact minimax regret over all 180,361 agent-turn histories for the size-control checkpoints: C-42 0.1646, C-TBR-43 0.1848, board-distilled 300-wide 0.1580, 600-wide s0 0.1551 / s1 0.1567, full-history 300-wide 0.0053. |
| `repair_C42_s0.json`, `repair_histfull_s0.json`, `repair_hist20_s0.json` | `src/repair.py` | Certificate-guided repair for C-42, the full-history GRU (`runs/distill_hist_C_histall_s1`), and the 20-epoch GRU (`runs/distill_hist_C_histall_s0`). Each file holds the base certificate, the repair log (certificate after every round, aggregate size, aggregate regret before/after the fit), the final aggregate histories, and six matched controls (three uniform over all 180,361 histories, three uniform over the 5,097-history optimal-play tree). C-42: 7 histories, 4 rounds, `W=0→1`, exact mean 25.0→100.0, history regret 0.1646→0.1713. Full-history: 5 histories, 2 rounds, 37.5→100.0, 0.00526→0.00538. Broken: 3 histories, 2 rounds, 0.0→100.0, 0.0346→0.0388. All 18 controls: `W=0`. The C-42 file also records DAgger arms under `dagger`: matched budget (7 states) certifies 0/3, full visited budget certifies 2/3 with 20 states, and the failing seed spends 72 states without certifying. |
| `worst_case_repair_C42.json`, `worst_case_repair_others.json` | `src/worst_case_eval.py` | Independent re-certification of the three saved repaired checkpoints with the original implementation: all three `W=B=1`, exact mean 100%, over 26, 40 and 26 memoized states. |
| `c4_canon_s0..2.json` | `src/c4_repair.py --rows 4 --cols 3 --win 3` | Exact Connect Three certificate and CGR for three seeds. The base GRU is trained on all 933 learner-turn boards (one canonical history each) and has zero board-level mistakes; base `W=0`, `B=1`, exact mean 11.1/22.2/33.3%. CGR adds 11/9/8 histories in 5/5/2 rounds and reaches `W=B=1` with zero replay mistakes. Six matched controls per seed stay at `W=0`. Seed 0 also records six DAgger arms: matched budget (11 states) certifies 2/3 seeds, full visited budget (27-32 states) certifies 2/3, and one seed in each variant fails. |
| `c4_verify_s0..2.json` | `src/c4_verify.py` | Independent memo-free re-certification of the repaired Connect Three checkpoints: `W=1`, all six deterministic column-order teachers draw, and 2,000 random adaptive optimal teachers draw 100%. |
| `c4x4_ctrl_s0.json` | `src/c4_repair.py --rows 4 --cols 4 --win 4 --replay teacher` | Connect Four on a 4x4 board (161,029 positions). The base GRU is trained on the 4,328-history tree of one deterministic optimal teacher (194 constrained); base `W=0`, `B=1`, mean 32.8%, 75 violations, 946 frontier states. CGR adds 1,182 histories in 33 rounds and reaches `W=B=1` with zero replay mistakes; the final certificate has 8,877 states and a 3,572-state frontier. Two matched controls add 1,181 histories with the same per-round budget and optimizer (104 and 68 epochs vs. 116 for the repair) and stay at `W=0`; the constrained-history control reaches mean 48.2%, the optimal-play-tree control 32.5%, both with more than 100 violations. |
| `c4x4_verify_s0.json` | `src/c4_verify.py --rows 4 --cols 4 --win 4` | Independent re-certification of the repaired 4x4 policy: `W=1`, all 24 deterministic column orders draw, 2,000 adaptive optimal teachers draw 100%. |
| `c4_end_ctrl_s0.json` | `src/c4_endgame.py --roots 40 --empties 8 --root-value 0` | Exact conditional certificates on 40 sampled 7x6 endgames with eight empty cells, all draws under optimal play. Base GRU (57k parameters) trained on 498 histories: `W=0`, `B=1`, mean 76.6%, 43 violations, 22/40 roots drawn. CGR adds 111 histories in 8 rounds, reaches `W=B=1`, and leaves one replay decision suboptimal. Six matched controls stay at `W=0` (mean 75.2-92.1%); the optimal-play-tree controls reach 87.6-92.1% on average without removing the worst case. |
| `c4_end_s1.json`, `c4_end_s2.json` | `src/c4_endgame.py --roots-file data/c4_endgame_roots_8_40_s0.json` | Replication of the 7x6 endgame repair on the **same** 40-root suite. Base mean 78.9%/79.4%, 42/40 violations, 25/23 roots drawn; CGR adds 102/95 histories in 6/7 rounds and reaches `W=B=1` in both seeds, leaving three/one replay decisions suboptimal. |
| `c4_end10_s0.json` | `src/c4_endgame.py --roots 40 --empties 10 --root-value 0 --agg-repeat 60 --stop both --max-iters 24` | The same 7x6 protocol two plies deeper: 40 draw roots with **ten** empty cells, 1,038 replay histories (182 constrained). Base GRU: `W=0`, `B=1`, mean 68.2%, 59 violations, 18/40 roots drawn. CGR adds 214 histories in 17 rounds, reaches `W=B=1` with mean 100% and zero replay mistakes; violations per certificate pass are 59, 39, 20, 19, 11, 15, 6, 9, 9, 3, 6, 5, 3, 5, 2, 2, 1, 0. The two matched controls (seed-1 optimizers, matched to the repair arm's per-round counts) add 182 (mean 73.0%, 49 violations) and 150 histories (mean 87.0%, 36 violations) and stay at `W=0`. Timing on a two-core laptop: 6,455 s for the repair arm, 41,192 s for the file including both controls. |
| `c4_end10_s1.json` | `src/c4_endgame.py --empties 10 --roots-file data/c4_endgame_roots_10_40_s0.json --no-controls` | First replication of the ten-empty-cell arm on the **same** 40-root suite: 1,151 replay histories, base `W=0`, `B=1`, mean 72.9%, 48 violations, 19/40 roots drawn. CGR adds 166 histories in 14 rounds and reaches `W=B=1` with mean 100%; violations per certificate pass are 48, 40, 19, 14, 15, 11, 2, 6, 1, 3, 1, 2, 3, 1, 0 while roots drawn climb 19→40. Repair loop 2,231 s, file 2,502 s on the two-core laptop. This arm was archived before base checkpoints were saved, so its two matched controls are not on disk yet: `--base-fit` re-derives its starting policy (verified against the recorded base certificate) and `--controls-from` then produces `c4_end10_s1_all.json` and `c4_end10_s1_tree.json`. |
| `c4_approx_s0.json` | `src/c4_approx_oracle.py --depths 0,1,...,12` | First empirical probe of the approximate-certificate proposition on Connect Three (4x3, horizon 12, 2,715 reachable boards). The oracle is depth-limited minimax whose leaves are the exact expectation of uniform random play; the table records its worst error against the exact oracle, the induced teacher class, and the repair run it drives. The error does not tune smoothly: with integer payoffs it is 1.34, 1.34, 1.13, 1.04 and then pinned at 1.00 for depths 4-7, 0.22 at depth 8, and 0 at depths 10-12, so no truncation reaches the `2*epsilon*H < 1` regime of the proposition (it needs `epsilon < 1/24`) and `bound_holds` is true only for the exact cases. Repair follows the *shape* of the class rather than the error size: depths 0-1 never certify in five rounds, depths 2-3 and 8 certify immediately with no added histories, and depths 4-7 - where an error of exactly 1 admits every action, i.e. every opponent - need 9-12 rounds and 81-109 histories. Conclusion: horizon truncation cannot test the tolerance threshold, and a continuously tunable oracle-error family is required. 1,041 s total, one core. |
| `rps_s0..2.json` | `src/rps_repair.py` | Hidden-type Rock-Paper-Scissors (H=8, six deterministic teacher types, uniform posterior, exact Bayes-adaptive DP over 19,234 histories). Base GRU trained on the 3,280-history COPY tree: exact value 2.667/2.833/2.667, certificate `V^=0.698/0.709/0.698`, 19/19/20 frontier violations, per-type scores 0-1 against the constant types but 8/8 against COPY and WSLS. CGR reaches `V^=1.000` in all three seeds (53/39/47 histories in 7/5/5 rounds, zero replay mistakes) and every final per-type vector is the Bayes optimum (6,7,6,8,7,8). Seed 0 records three random control pools and two DAgger variants with three seeds each, matched per-round: COPY-tree 0.709, all-histories 0.748, optimal-tree 1.000 (its 34 histories are a whole frontier and fit the budget), DAgger-all 0.977 with 1/3 certified, DAgger-worst 0.969. Each file's `repair.aggregate` stores the selected histories themselves. |
| `c4_end_verify_s0.json` | `src/c4_endgame_verify.py` | Independent memo-free verification of the repaired 7x6 endgame policy: `W=B=1`, 120 deterministic column orders draw all 40 roots, 40 adaptive optimal games per root draw 100%. |
| `c4_endgame_roots_8_40_s0.json` | `src/c4_endgame.py` | The 40 serialized endgame roots (prefix, board, exact value) so the certificate can be replayed independently. |

## Legacy v1 artifacts (kept, not used as v2 claims)

| File | Produced by | Status |
|---|---|---|
| `twin_mentor.json` | `src/twin_mentor.py` | First twin-gap measurement (one checkpoint/config, one twin seed). Superseded by `tbr_eval_*` and `exact_twin_*`. |
| `v1_reanalysis_A/B/C/D.json` | `src/tbr_eval.py` | 200-game re-evaluation of retrained v1 checkpoints with 3 twins. A/B/C feed Table 1; D is not used. |
| `ablation_results.json`, `cross_eval_results.json` | `src/ablation_ttt.py`, `src/evaluate_cross.py` | Training and cross-evaluation metrics. The `*_random` key is **not** uniform random; it is a greedy depth-0 solver. Corrected uniform-random wins are A 65.5%, B 55.0%, C 71.0%, D 80.0%. |
| `benchmark_act_results.json` | `src/benchmark_act.py` | ACT wall-clock benchmark from one Windows CPU. The 2.45× overhead did not replicate (0.80× rerun; per-seed 521–4,283 s). Not a v2 claim. |
| `tournament_results.json` | `src/tournament.py` | Round-robin directed win rates. The reported GRU > CfC > ACT cycle is retracted (seed variance). |
| `transfer_results.json` | `src/transfer_connect4.py` | Tic-tac-toe → Connect Four transfer. 24/30 vs 7/30 is numerically reproducible but single-checkpoint; reported as suggestive only. |
| `fragility_curve.json`, `latent_geometry.json`, `legibilidad.json`, `equivariance.json`, `proposito.json`, `multiseed_summary.json`, `distilacion_mentor.json` | respective scripts | Exploratory v1 diagnostics. Their effective sample sizes are 1–7 unique states (or one repeated line); the latent-geometry, equivariance and legibility claims are unsupported and excluded. |

`_*.json` and `_smoke_*.json` are smoke runs used to verify scripts, not paper
evidence.

## Known limitations

1. **Game counts are not independent observations.** A deterministic agent
   against a deterministic teacher repeats one game. Report unique boards and
   histories, not only games; the paper uses 5–13 histories as the effective
   certification sample.
2. **Deterministic action selection.** All reported evaluations use
   `deterministic=True`. Distinct checkpoints can coincide on the evaluated
   paths; C-43 and C-44 do, even though their zero-state policies differ on 783
   of 2,423 boards. The nine table rows are therefore not nine independent
   behaviors.
3. **Few seeds.** Three RL seeds per condition and two MLP seeds. With n=3 the
   exact two-sided sign-flip test cannot reach p<0.05; TBR effects are reported
   as unresolved, not significant.
4. **Oracle validity.** `Oracle.lookup` mirrors the board and is valid only on
   agent-turn boards. Teacher-turn depth-6 vs. depth-9 equivalence was verified
   separately on all 2,097 reachable teacher-turn boards; do not use
   `Oracle.lookup` there.
5. **Checkpoints are not versioned.** `sha256_16` ties a data row to weights,
   but if a checkpoint is retrained the row cannot be reproduced from the file
   name alone.
6. **Legacy files mix runs.** `ablation_results.json` predates the current
   `*.pth` files, and the root and `data/` copies are not byte-identical. Do not
   pair training metrics with checkpoint evaluations without retraining.
7. **Single-machine timing.** Latency numbers are wall-clock from one machine;
   the CPU model is not recorded.
8. **Absolute paths.** Some `args` blocks embed the packaging machine's paths
   (`C:\Users\...`); they are provenance, not portable configuration.
9. **Certificate scope.** `worst_case.json` is exact only for deterministic
   learners with exact values. The adversarial teacher may condition its
   tie-breaks on the full history, which is inside the optimal class. For
   approximate ties the certificate is relative to the tolerance used to define
   the optimal set.
10. **Average history regret is not the certificate.** `history_metrics_controls.json`
    averages regret over all 180,361 histories, including histories no optimal
    teacher can reach. C-TBR-43 is certified `W=B=1` with mean history regret
    0.185, while the full-history GRU has 0.005 and is not certified. Use
    `worst_case_eval.py` for robustness claims and the history metric only as an
    average-case diagnostic.

11. **Repair assumes the aggregate can be fitted.** CGR's progress guarantee is
    conditional on each round fitting the violating histories exactly; the runs
    check this condition (aggregate regret 0) but the theory does not prove it
    for every model. The final certificate is exact regardless. Repair can also
    lower average accuracy off the frontier: C-42's all-history regret rises
    from 0.1646 to 0.1713 while its worst case goes from 0 to 1.
