# Certify, Then Repair

**Exact worst-case guarantees for recurrent policies trained against strong teachers.**

[Paper (PDF, 22 pages)](paper/paper_v2.pdf) · [Source](paper/paper_v2.tex) · [Data card](docs/DATA_CARD.md) · [Reproduction guide](docs/REPRODUCING.md) · [DOI (all versions)](https://doi.org/10.5281/zenodo.21077401)

A CPU-only study of game-playing agents (CfC and GRU, trained with REINFORCE +
GAE against minimax teachers) that asks a simple question: when we certify a
policy against a perfect teacher, what exactly have we certified?

## The result in one paragraph

Minimax is a *value*, not a *policy*. At 44.7% of teacher decisions in this game
the optimal move is not unique (mean 1.95 optimal moves; up to 9 at the opening).
We train nine recurrent checkpoints that draw **100%** of games against a fixed
deterministic depth-6 teacher, then evaluate them against **twins**: teachers with
the same depth and the same value function that differ only in how they break
ties. The median draw rate falls to **24.3%** — a **75.8-point** collapse — and the
same collapse appears against an exact depth-9 oracle (median 24.9%). The gap is
not a depth-6 artifact, not teacher noise, and not fixable by randomizing ties
during training. It comes from what the policy's recurrent state encodes: the
ordered history of play, not the board. A memoryless MLP over the 9-cell board
closes the gap completely.

An exact backward-induction certificate over **all** optimal teachers sharpens
the claim: every fixed-trained checkpoint has **worst-case draw rate 0** and
**best-case 1**. Its 100% exact-first result is one extreme of the interval, not
the typical case. The certificate visits 27–90 states (vs. 180,361 histories),
returns a six-ply exploit against C-42, and certifies only the memoryless MLP
and one of the six TBR runs.

The certificate is also a repair map. A policy that is optimal on its own
optimal-play frontier cannot lose, so the frontier histories are a sufficient
training set. **Certificate-Guided Repair** (CGR) adds only the violating
frontier histories, fits the same-size model again, and re-certifies: **7**
histories take C-42 from `W=0` to `W=1`, **5** repair the GRU trained on all
180,361 histories, and **3** repair a model that loses to every optimal teacher.
Matched random aggregation with the same number of states never certifies, and
matched DAgger is unreliable: zero of three Tic-Tac-Toe seeds, two of three
Connect Three seeds. The same repair works when the teacher's type is hidden and
the state is a belief: in Rock-Paper-Scissors an exact Bayes-adaptive certificate
finds 39–53 states that take a COPY-trained GRU from `V^≈0.70` to the Bayes
optimum in all three seeds.

## Headline evidence

| Checkpoint (400 games/opponent) | vs fixed M6 | vs 3 random twins | Gap | vs exact twins | Exact gap |
|---|---:|---:|---:|---:|---:|
| B-42 CfC×3 | 100.0% | 24.9% | −75.1 | 24.9% | −75.1 |
| B-43 CfC×3 | 100.0% | 34.8% | −65.3 | 37.2% | −62.8 |
| B-44 CfC×3 | 100.0% | 51.3% | −48.7 | 50.7% | −49.3 |
| C-42 GRU | 100.0% | 23.9% | −76.1 | 23.9% | −76.1 |
| C-43 GRU | 100.0% | 25.0% | −75.0 | 26.5% | −73.5 |
| C-44 GRU | 100.0% | 24.3% | −75.8 | 26.5% | −73.5 |

All nine checkpoints (including the three v1 A/B/C models): **100%** vs fixed M6,
median **24.3%** vs twins (range 20.5–51.3), median gap **75.8** points. The exact
depth-9 deterministic teacher gives **100% draws and zero regret** for every
checkpoint, so the fixed-teacher certification is not wrong — it is incomplete.

The worst case is not noise: the exact-twin control reproduces the same range
(20.5–50.7%, median 24.9%), and twin regret is 0.12–0.21 while fixed-teacher
regret is 0.000.

## Why: state is history, not the board

Tic-tac-toe has only **2,423** reachable agent-turn boards. The game tree
contains **180,361** distinct reachable agent-turn *histories* — a 74.4× blow-up.
Of those histories, 25,505 are tie histories (14.1%). History-level regret is
therefore a different object from board-level regret:

- A board-level canonical probe gives 0.518 regret / 0.662 optimal rate.
- The history-level probe on the same checkpoint gives 0.165 / 0.896.

Certification inherits this: a 400-game evaluation against the fixed teacher
visits **5 boards / 5 histories** (0.003% of history space); a 3,000-game,
30-twin evaluation visits **12 boards / 13 histories** (0.007%). The apparent
success is one line of play.

Coverage behaves as a dose–response: board regret is 0.49 at 25 distilled boards,
0.31 at 400, 0.16 at 1,200, and 0.000 at all 2,423 — yet the full-board recurrent
model still draws only **32.2%** against twins. Data is not the binding
constraint; the state representation is.

## The constructive fix

A memoryless MLP (9–256–256–9) trained on soft exact-oracle targets — no
recurrence, no history — achieves:

- **100%** board-optimal decisions and 0.000 regret over 2,000 decisions,
- **400/400** draws against the fixed teacher and **60/60** held-out teachers at
  100% (30 random-tie twins + 30 episode-coherent twins, 100 games each),
- 97.75% / 95.75% wins against uniform random, 100% against a weak minimax.

Feeding the same oracle targets to a *recurrent* model does not fix it: 0.000
regret on canonical training histories, but 0.158 history regret, 90.2% optimal
rate, and 32.25% twin draw rate. **Representation, not data, is the difference.**

## Tie-break randomization is a lottery, not a fix

Training with random tie-breaking (TBR) can move the failure rather than remove
it. Across six TBR runs:

| Run | M6 | 3 twins | 30-twin mean [range] | Status |
|---|---:|---:|---:|---|
| C TBR-43 | 100% | 100% | 100% [100,100] | robust |
| C TBR-42 | 100% | 49.8% | 51.3% [38,59] | partial |
| B TBR-42 | 0% | 68.5% | — | relocated |
| C TBR-44 | 0% | 75.2% | 75.6% [70,84] | relocated |
| B TBR-43 | 0% | 0% | — | fail |
| B TBR-44 | 0% | 0% | — | fail |

Mixed and episode-coherent teachers cap out near 50%; teacher noise has no
effect (23.9/24.0% vs 23.9/25.0% fixed). The paired TBR effect is not
statistically resolved (C: +0.506, p=0.25; B: −0.142, p=0.75), and with three
seeds no two-sided p<0.05 is attainable. TBR is a seed lottery whose best draw
is a genuinely twin-robust policy — not a training recipe.

## Exact certificate: 100% is the best case, not the worst

Every sampled twin ensemble estimates an average. Because the game is solved we
can search the entire set of deterministic optimal teachers exactly and compute
the worst case `W`, the best case `B`, and the exact uniform mean `D̄`:

| Policy | `W` | `B` | exact `D̄` | 3 exact-twin sample |
|---|---:|---:|---:|---:|
| nine fixed checkpoints | 0 | 100 | 25.0 (B-43 37.5, B-44 50.0) | 20.5–50.7 |
| C TBR-43 | **100** | 100 | 100 | 100 |
| C TBR-44 | 0 | 100 | 75.0 | 75.2 |
| recurrent distill, full coverage | 0 | 100 | 33.3 | 32.2 |
| memoryless MLP (both seeds) | **100** | 100 | 100 | 100 |

For every fixed checkpoint there is an exactly optimal teacher that beats it and
another that draws every game. The certificate also emits the exploit:
against C-42, `O4 X2 O3 X5 O0 X8` — the first four moves are optimal, O0 is the
agent's only mistake (exact regret 1), and X8 completes the right column. A
board-optimal memoryless policy is worst-case robust by construction; a policy
that is board-optimal only on canonical histories is not. The recursion is
27–90 memoized `(board, history)` states.

## Is it model size?

No. Four exact controls separate parameter count from the teacher and the state:

| Model | Params | History regret | `W` | `B` |
|---|---:|---:|---:|---:|
| C-42 (GRU 300, fixed ties) | 323,491 | 0.165 | 0 | 100 |
| C-TBR-43 (GRU 300, random ties) | 323,491 | 0.185 | **100** | 100 |
| GRU 600 (two seeds, same data) | 1,185,691 | 0.155/0.157 | 0 | 100 |
| GRU 300, all 180,361 histories | 323,491 | 0.005 | 0 | 100 |
| MLP 256 (memoryless) | 70,665 | 0.000 | **100** | 100 |

A memoryless model with **4.6× fewer** parameters is certified; a GRU with
**3.7× more** is not, even though both fit their training set perfectly; the
same 323k GRU is certified (C-TBR-43, one of six TBR runs) when only the mentor's tie-break changes. Complete
history coverage cuts mean regret 30-fold but still leaves `W=0`, and C-TBR-43
is certified with *higher* average history regret than C-42, because its errors
lie off the tree that optimal teachers actually visit. Average regret over the
history space is neither necessary nor sufficient for robustness.

## Certificate-guided repair (CGR)

The exact certificate does not only answer *whether* a policy is robust; it
returns the exact learner-turn histories on which the answer depends (the
frontier reachable under all optimal tie-breaks). A policy that is optimal on
that frontier cannot lose. CGR therefore alternates certificate and repair:
certify, add the violating frontier histories with exact oracle targets, fit the
same-size model, re-certify.

| Target | Before `W/B/D` | Added | Rounds | After `W/B/D` | History regret |
|---|---|---:|---:|---|---:|
| C-42 (RL, fixed ties) | 0/100/25.0 | **7** | 4 | **100/100/100.0** | 0.165 -> 0.171 |
| GRU 300, all 180,361 histories | 0/100/37.5 | **5** | 2 | **100/100/100.0** | 0.0053 -> 0.0054 |
| GRU 300, 20-epoch distillation (W=B=0) | 0/0/0.0 | **3** | 2 | **100/100/100.0** | 0.0346 -> 0.0388 |

The control adds the same number of histories per round, with the same optimizer,
epochs and fitting protocol, sampled without looking at violations --- uniformly
from all 180,361 histories or from the 5,097-history optimal-play tree. None of
the 18 controls reaches `W=1`; several improve the average while leaving the
worst case at zero. Selecting the states, not adding states, is what repairs the
policy.

A DAgger control with the same certificate and the same stopping rule is
competitive but unreliable. Given CGR's seven-state budget it fails in all three
Tic-Tac-Toe seeds; given every visited state it certifies two of three, needing
20 states, while the third seed adds 72 --- ten times CGR's budget --- and still
misses a violating history. Visiting the learner's own states is not the same as
visiting the states its test names.

The three repaired checkpoints are re-certified by the independent
`worst_case_eval.py` implementation at `W=B=1` over 26, 40 and 26 states, so the
result does not depend on the repair script's own recursion. The paper proves
frontier sufficiency, the soundness and progress of the loop, and a tolerance
version for approximate oracles (greedy play on an eps-frontier loses at most
`2*eps*H`, and loses nothing at all once `2*eps*H` falls below the spacing
`Delta` of the game's values), and reports the small average-regret price of the
exact guarantee. That tolerance is measured rather than assumed: degrading the
Connect Three oracle by blending it toward random play shows the certified
teacher class opening exactly at `eps = Delta`, the learner's argmax misranking
from `Delta/2`, certification surviving 14x (integer payoffs) to 133x (graded)
past the threshold the proof needs, and the bound 19-36x looser than the
observed deficit. The failure it predicts is the one that is easy to miss: in 30
of the 32 arms that never certify, the loop ends with **no violating history
left to add** and the policy still loses. `src/c4_approx_oracle.py` produces
those numbers, `--gaps-only` writes the payoff-spacing ladder they are read
against, and `src/eps_figure.py` plots them.

## A second domain: Connect Three (4x3)

The failure is not specific to Tic-Tac-Toe. On a 4x3 Connect Three board the
exact oracle enumerates all 2,715 reachable boards, of which 933 are
learner-turn positions. The base policy (a 14,443-parameter GRU) is trained on
one canonical history for **every** one of those 933 boards and reaches zero
board-level mistakes. It is still fragile: the exact certificate over all
optimal teachers gives `W=0`, `B=1`, and `D=11.1/22.2/33.3%` for seeds 0-2,
and it loses to every deterministic optimal teacher in seeds 0-1.

Every violating history is an **alternative history to a board that the policy
already plays optimally**. This isolates the history dependence: board coverage
is complete and the worst case is still zero.

| Seed | Before `W/B/D` | Violations | Added | Rounds | After `W/B/D` | Replay mistakes |
|---|---|---:|---:|---:|---|---:|
| 0 | 0/100/11.1 | 6 | **11** | 5 | **100/100/100.0** | 0 |
| 1 | 0/100/22.2 | 5 | **9** | 5 | **100/100/100.0** | 0 |
| 2 | 0/100/33.3 | 4 | **8** | 2 | **100/100/100.0** | 0 |

The repair stops only when both the 933-board replay and the selected histories
are exactly optimal, so board accuracy is preserved. The independent memo-free
verifier (`c4_verify.py`) re-certifies each saved checkpoint at `W=1`: all six
deterministic column-order teachers draw and 2,000 randomly adaptive optimal
teachers draw 100%. All 18 matched controls stay at `W=0`; their averages never
exceed the largest base average (33.3%).

Matched DAgger is the boundary case in this game: it certifies two of three seeds
at the same 11 states, and with every visited state it also certifies two of
three at 27-32. The frontier here has five states, small enough for visitation to
hit it; the Tic-Tac-Toe frontier has thirteen and matched DAgger never does.

```bash
python src/c4_repair.py --rows 4 --cols 3 --win 3 --hidden 64 --seed 0 \
  --epochs 300 --lr 2e-3 --max-iters 10 --control-seeds 1,2,3 \
  --control-mode all,tree --ckpt-out runs/c4_canon_s0 --out data/c4_canon_s0.json
python src/c4_verify.py --ckpt runs/c4_canon_s0/c4_gru.pt --hidden 64 \
  --games 2000 --out data/c4_verify_s0.json
python src/c4_figure.py --out figs_v2/fig7_c4.pdf
```

`make c4` runs all three seeds, the verifier, and the figure.

## Scaling: Connect Four (4x4) and 7x6 endgames

**Connect Four, 4x4** (161,029 reachable positions). The base GRU is trained on
the 4,328-history tree of one deterministic optimal teacher (194 constrained
states), reaches zero replay mistakes, and is still certified at `W=0`,
`B=1`, `D=32.8%` with 75 violating histories. CGR adds **1,182** histories over
**33** rounds and reaches `W=B=1`, with zero replay mistakes; the certificate
grows from 2,307 to 8,877 states. The independent verifier gives `W=1`, 24/24
deterministic orders draw, and 2,000 adaptive optimal teachers draw 100%.
Two matched controls add 1,181 histories with the same per-round budget and
optimizer (104 and 68 optimizer epochs against 116 for the repair arm) and stay
at `W=0`; the constrained-history control raises the mean to 48.2%, while the
optimal-play-tree control ends at 32.5% with 117 violations.

**7x6 endgames** (40 roots with eight empty cells, all draws under optimal
play; 498 training histories). The three seeds share the same root suite. The
base 57k-parameter GRUs are exact on all 498 replay histories and still
certified at `W=0`, `B=1`, with `D=76.6/78.9/79.4%` for seeds 0/1/2 (43, 42,
and 40 violations; 22, 25, and 23 of 40 roots drawn). CGR adds **111, 102, and
95** histories in **8, 6, and 7** rounds and reaches `W=B=1` in all three
seeds; one, three, and one of the 498 replay decisions regress. The
independent verifier gives `W=1` for seed 0: 120 deterministic column orders
draw all 40 roots, and 40 adaptive games per root draw 100%. Six matched
controls on seed 0 stay at `W=0` (mean 75.2-92.1%); the optimal-play-tree
controls reach 87.6-92.1% on average yet still lose at least one root. This
certificate is exact **conditional on the sampled endgame suite**, not for the
full 7x6 game.

**Ten empty cells on the same board** (same 40-root suite shape, deeper
endgames; 1,038 training histories). Depth, not board size, is the scaling
axis. The base GRU is exact on its replay set and still certified at `W=0`,
`B=1`, `D=68.2%` with 59 violations and 18/40 roots drawn. CGR needs **17
rounds and 214** certificate-selected histories to reach `W=B=1`, `D=100%`
(eight cells needed 6-8 rounds and 95-111 histories); this arm repeats each
added history 60 times and stops a refit only when replay and aggregate are
both exactly optimal. Both matched controls fail with more data than the
eight-cell controls used: constrained-pool aggregation with 182 histories ends
at `W=0`, `D=73.0%` (49 violations) and optimal-play-tree aggregation with 150
histories ends at `W=0`, `D=87.0%` (36 violations). The two replicas on the
identical suite both certify too, at very different prices: seed 1 (1,151
histories) starts from 48 violations and needs **14 rounds and 166** histories,
2,502 s of one two-core laptop; seed 2 (1,089 histories) starts from 49 and needs
**7 rounds and 166** histories, 2,244 s. Rounds for the same exact procedure
therefore range from 7 to 17 at this depth, while all three end at `W=B=1` with
zero replay mistakes. A memo-free verifier re-certifies all three checkpoints at
`W=B=1`: 120 deterministic column orders and 40 adaptive optimal teachers per
root draw every root.

```bash
python src/c4_repair.py --rows 4 --cols 4 --win 4 --replay teacher --hidden 64 \
  --epochs 120 --lr 2e-3 --agg-repeat 10 --stop agg --max-iters 40 --seed 0 \
  --control-seeds 1 --control-mode all,tree --control-epochs 10 \
  --ckpt-out runs/c4x4_ctrl_s0 --out data/c4x4_ctrl_s0.json
python src/c4_verify.py --ckpt runs/c4x4_ctrl_s0/c4_gru.pt --rows 4 --cols 4 \
  --win 4 --hidden 64 --games 2000 --out data/c4x4_verify_s0.json

python src/c4_endgame.py --roots 40 --empties 8 --root-value 0 --branch 3 \
  --hidden 128 --emb 16 --epochs 200 --agg-repeat 20 --stop agg --max-iters 12 \
  --seed 0 --control-seeds 1,2,3 --control-mode all,tree \
  --ckpt-out runs/c4_end_ctrl_s0 --out data/c4_end_ctrl_s0.json
# seeds 1-2 reuse the same root suite so the per-seed comparison is matched
for s in 1 2; do
  python src/c4_endgame.py --roots-file data/c4_endgame_roots_8_40_s0.json \
    --roots 40 --empties 8 --root-value 0 --branch 3 --hidden 128 --emb 16 \
    --epochs 200 --agg-repeat 20 --stop agg --max-iters 12 --seed $s \
    --no-controls --ckpt-out runs/c4_end_s$s --out data/c4_end_s$s.json
done
python src/c4_endgame_verify.py --ckpt runs/c4_end_ctrl_s0/c4_gru.pt \
  --roots data/c4_endgame_roots_8_40_s0.json --hidden 128 --emb 16 \
  --orders 120 --games 40 --out data/c4_end_verify_s0.json

# ten empty cells: seed 0 with both matched controls, seeds 1-2 as replicas
python src/c4_endgame.py --empties 10 --roots 40 --root-value 0 --branch 3 \
  --hidden 128 --emb 16 --epochs 200 --agg-repeat 60 --stop both --max-iters 24 \
  --seed 0 --control-seeds 1 --control-mode all,tree \
  --ckpt-out runs/c4_end10_s0 --out data/c4_end10_s0.json
for s in 1 2; do
  python src/c4_endgame.py --roots-file data/c4_endgame_roots_10_40_s0.json \
    --empties 10 --roots 40 --root-value 0 --branch 3 --hidden 128 --emb 16 \
    --epochs 200 --agg-repeat 60 --stop both --max-iters 24 --seed $s \
    --no-controls --ckpt-out runs/c4_end10_s$s --out data/c4_end10_s$s.json
done
```

Twelve empty cells and beyond are compute-bound rather than algorithm-bound:
the certificate is still exact, but a two-core laptop needs hours per seed.
[`colab/c4_endgame_scaling.ipynb`](colab/c4_endgame_scaling.ipynb) runs a whole
depth on a Colab CPU instance as a queue of separate processes - three repair
arms, the two matched controls of the reference seed, three verifications - one
per usable core, over a single shared 40-root suite. CPU is the right choice,
not a compromise: the certificate is a pure-Python recursion and the policy is a
57k-parameter GRU, so a GPU would accelerate neither. The split is exact and
pinned by a test: `--controls-from` reproduces the per-round certificate logs of
the inline arms it replaces, and `--base-fit` re-derives the starting policy of an
arm archived before checkpoints were saved.

The queue needs no terminal and no token. The setup cell signs the runtime in to
GitHub with the device flow - it prints a one-time code you type on your phone at
`github.com/login/device`, the credential lives in that runtime and dies with it,
and the repository stays private - and if the clone still fails the cell offers a
token, a Drive zip of the repo, or a public repo as alternatives. A hands-free
cell then pumps the queue every two minutes until it empties or five hours pass;
every finished arm is mirrored to Drive as it lands, so an interrupt or a session
cap costs the arms in flight and nothing already done.

## A third domain: hidden-type Rock-Paper-Scissors

The failure does not require a solved game or perfect information. In an
eight-round Rock-Paper-Scissors game against one of six hidden deterministic
teachers (always Rock, Paper, Scissors, COPY, COUNTER, or win-stay/lose-shift)
the learner observes actions but never the type. The state is the history and
the exact object is the Bayes-adaptive value under the uniform posterior over
consistent types: 19,234 reachable histories, `V*=7.000`, `Vmin=-7.333`.

The base GRU is trained on the 3,280 learner histories of the COPY tree. On that
tree the teacher's action after round *k* is the learner's own action at round
*k-1*, so the teacher channel is redundant given the learner tokens: the network
can fit all 3,280 decisions while ignoring it, and it does. The base scores 8/8
against COPY and WSLS but 0-1 against the constant types, and the exact
certificate is `V^=0.698/0.709/0.698` (seeds 0/1/2) with 19/19/20 violating
histories.

CGR with the same protocol (add the violating frontier histories with exact
targets, oversample 40×, refit, re-certify) reaches the Bayes optimum `V^=1` in
all three seeds: 53, 39, and 47 histories in 7, 5, and 5 rounds, zero replay
mistakes, final per-type vector exactly `(6,7,6,8,7,8)`. Each trajectory is
non-monotone (seed 2 even dips from 0.698 to 0.651 after the first refit), and
the certificate catches every regression at the next round. The independent
verifier re-certifies each checkpoint at `V^=1` (11/11 checks).

| Seed | Before `V^` | Violations | Added | Rounds | After `V^` |
|---|---|---:|---:|---:|---:|
| 0 | 0.698 | 19 | 53 | 7 | **1.000** |
| 1 | 0.709 | 19 | 39 | 5 | **1.000** |
| 2 | 0.698 | 20 | 47 | 5 | **1.000** |

Matched baselines (seed 0, three seeds each, same per-round added counts,
optimizer, epoch budget, and exact stopping rule):

| Baseline | Added | Final `V^` mean [min, max] | Certified |
|---|---:|---|---:|
| Random, COPY tree (3,280) | 53 | 0.709 [0.709, 0.709] | 0/3 |
| Random, all histories (19,234) | 53 | 0.748 [0.709, 0.779] | 0/3 |
| Random, Bayes-optimal tree (34) | 34 | 1.000 [1.000, 1.000] | 3/3 |
| DAgger, all six types | 53 | 0.977 [0.965, 1.000] | 1/3 |
| DAgger, worst type | 43 | 0.969 [0.965, 0.977] | 0/3 |
| **CGR** | **53** | **1.000** | **3/3** |

What matters is sampling states the learner can actually reach—its frontier.
Random pools do not (0.71-0.78); DAgger and the optimal tree do, and at 34
states the whole optimal tree fits in the budget. The same control fails in
Tic-Tac-Toe, where the optimal-play tree has 5,097 histories. The tree's 34
states visit 7 of the 12 beliefs in the game, including all four at which the
base errs. CGR is the only arm that both samples the learner's reachable states
and certifies every seed, but at this scale DAgger is not distinguishable from
it in sample efficiency—CGR's edge is that its states are the ones the
certificate names, so the same budget closes the exact test.

```bash
python src/rps_repair.py --seed 0 --max-iters 12 --threads 2 \
  --ckpt-out runs/rps_s0 --out data/rps_s0.json      # CGR + controls + DAgger
for s in 1 2; do
  python src/rps_repair.py --seed $s --max-iters 12 --threads 2 \
    --no-controls --no-dagger --ckpt-out runs/rps_s$s --out data/rps_s$s.json
  python src/rps_verify.py --data data/rps_s$s.json --ckpt runs/rps_s$s/rps_gru.pt
done
python src/rps_verify.py --data data/rps_s0.json --ckpt runs/rps_s0/rps_gru.pt
python src/rps_figure.py --seeds 0,1,2 --out figs_v2/fig9_rps.pdf
```

`make rps` runs the whole sequence, the verifier, and the figure;
`python -m pytest tests/test_rps.py -q` checks the oracle by brute force.

## Corrections to the v1 record

The repository keeps the old artifacts but does not repeat their claims:

| v1 claim | Status |
|---|---|
| 100% draw vs optimal M6 proves optimal play | **Ceiling only** — a random twin gets 24% |
| Agents collapse against stochastic play | **Retracted** — the “random” opponent was a greedy depth-0 minimax; uniform random wins are A 65.5%, B 55.0%, C 71.0%, D 80.0% |
| ACT costs 2.45× at inference | **Not robust** — rerun 0.80×, per-seed 521–4,283 s |
| GRU > CfC > ACT non-transitivity cycle | **Retracted** — seed variance swamps it |
| Tic-tac-toe skills transfer to Connect Four | **Suggestive only** — 24/30 vs 7/30, single checkpoint |
| Latent-geometry diagnostics (n_eff 5–7) | **Unsupported** — effective sample sizes too small |

## Install and test

Python ≥ 3.10 (tested 3.11.9), CPU only.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest tests -q      # 35 tests, ~2.5 min
make smoke                     # imports + CLI help + tests
```

## Reproduce the main results

Train a fixed-teacher and a TBR checkpoint (5,000 games):

```bash
python src/ablation_ttt.py --configs C --games 5000 --seed 42 \
  --tiebreak first  --logs-dir runs/fixed_C_s42 --results-file runs/fixed_C_s42/results.json
python src/ablation_ttt.py --configs C --games 5000 --seed 42 \
  --tiebreak random --logs-dir runs/tbr_C_s42   --results-file runs/tbr_C_s42/results.json
```

Evaluate against fixed and twin teachers (depth-6 and exact depth-9):

```bash
python src/tbr_eval.py \
  --runs "C_s42=runs/fixed_C_s42" --config C --games 400 \
  --twin-seeds 1234,5678,9012 --seed 20261027 --out data/tbr_eval_C_s42.json

python src/exact_twin_eval.py \
  --runs "C_s42=runs/fixed_C_s42" --config C --games 400 \
  --twin-seeds 1234,5678,9012 --seed 20261027 --out data/exact_twin_C_s42.json
```

State-space and coverage probes:

```bash
python src/coverage_probe.py --config C --ckpt runs/fixed_C_s42 --games 400 --out data/coverage.json
python src/twin_ensemble.py --config C --ckpt runs/fixed_C_s42 --seeds 30 --games 100 --out data/ensemble.json
python src/history_probe.py --config C --runs "C_s42=runs/fixed_C_s42" --out data/history_probe.json
python src/space_probe.py   --config C --runs "C_s42=runs/fixed_C_s42" --out data/space_probe.json
```

Exact worst-case certificate over all optimal teachers (21 recurrent checkpoints
plus both MLPs, ~3 minutes on CPU):

```bash
python src/worst_case_eval.py --preset all \
  --mlp "MLP-s0=runs/memoryless_s0/mlp.pt;MLP-s1=runs/memoryless_s1/mlp.pt" \
  --out data/worst_case.json
```

Optional 20,000-game Monte Carlo check of the exact uniform mean:

```bash
python src/worst_case_eval.py \
  --runs "B-v1=B:artifacts/v1_reentreno;C-42=C:runs/fixed_C_s42" \
  --mc-games 20000 --mc-seed 2026 --no-witness --out data/worst_case_mc_check.json
```

Memoryless control and recurrent distillation:

```bash
python src/memoryless.py --seeds 0,1 --out data/memoryless.json
python src/distill_oracle.py --config C --coverage 2423 --seeds 0 --out data/distill_C.json
python src/eval_memoryless.py --ckpt runs/memoryless_s0/mlp.pt --games 100 --twins 30 --coherent 30
```

Model-size controls (full-history distillation, 3.7× width, and history-level
regret for any checkpoint):

```bash
python src/distill_history.py --coverage all --epochs 20 --threads 4 \
  --out data/distill_history_C_full.json
python src/distill_history.py --coverage all --epochs 60 --lr 1e-3 --seed 1 \
  --init runs/distill_hist_C_histall_s0/ablation_C.pth --max-minutes 45 \
  --out data/distill_history_C_full_ft.json
python src/distill_oracle.py --config C --coverage 2423 --seeds 0,1 --epochs 400 \
  --hidden 600 --max-minutes 30 --out data/distill_C_h600.json
python src/worst_case_eval.py \
  --runs "C-h600-s0=C@600:runs/distill_C_cov2423_s0_h600" \
  --no-witness --out data/worst_case_h600_s0.json
python src/distill_history.py --eval-only --ckpt runs/fixed_C_s42/ablation_C.pth \
  --out data/history_metrics_controls.json
```

Certificate-guided repair (three failing checkpoints, matched random controls,
all-history evaluation; a few minutes each on CPU):

```bash
python src/repair.py --config C --ckpt runs/fixed_C_s42/ablation_C.pth --seed 0 \
  --control-seeds 1,2,3 --control-mode all,tree --eval-all \
  --ckpt-out runs/repair_C42_s0 --out data/repair_C42_s0.json
python src/repair.py --config C --ckpt runs/distill_hist_C_histall_s1/ablation_C.pth \
  --control-seeds 1,2,3 --control-mode all,tree --eval-all \
  --ckpt-out runs/repair_histfull_s0 --out data/repair_histfull_s0.json
python src/repair.py --config C --ckpt runs/distill_hist_C_histall_s0/ablation_C.pth \
  --control-seeds 1,2,3 --control-mode all,tree --eval-all \
  --ckpt-out runs/repair_hist20_s0 --out data/repair_hist20_s0.json
python src/worst_case_eval.py --runs "C-42-repaired=C:runs/repair_C42_s0" \
  --out data/worst_case_repair_C42.json
```

Rebuild figures and the paper (Tectonic at `.tools/tectonic.exe`):

```bash
python src/make_figures.py
make paper        # .tools/tectonic.exe paper/paper_v2.tex
```

Exact flags, artifact schemas and provenance are in
[docs/REPRODUCING.md](docs/REPRODUCING.md) and [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Repository layout

```
src/
  liquid_rl_trainer.py   CfC/GRU agents, environment, REINFORCE+GAE
  ablation_ttt.py        A–D training; --seed, --tiebreak, --configs
  minimax_mentor.py      Minimax, StochasticMinimax, tie-break policies
  oracle.py              Exact depth-9 oracle over reachable boards
  tbr_eval.py            Fixed/twin evaluation with Wilson CIs, regret, optimal rate
  exact_twin_eval.py     Exact depth-9 teacher/twin control
  twin_ensemble.py       30-seed twin ensembles (random vs episode-coherent)
  coverage_probe.py      Board/history coverage of a certification run
  history_probe.py       History-level canonical probe
  space_probe.py         Board-level canonical probe
  distill_oracle.py      Oracle distillation at controlled board coverage
  distill_history.py     Full-history distillation + history-level regret probe
  worst_case_eval.py     Exact worst/best/average over all optimal teachers + witness
  repair.py              Certificate-guided repair (CGR): frontier aggregation + controls
  repair_figure.py       Repair-trajectory figure from data/repair_*.json
  c4_game.py             Connect Three/Four oracle, canonical replay and certificate
  c4_repair.py           CGR on Connect Three/Four with matched controls
  c4_verify.py           Memo-free verifier for C4 checkpoints
  c4_endgame.py          7x6 endgame suite solver and conditional CGR
  c4_endgame_verify.py   Endgame verifier over sampled roots
  c4_approx_oracle.py    Depth-limited oracle with exact random-playout leaves,
                         for measuring how much teacher error a repair survives
  c4_figure.py           Connect Three/Four repair figure
  rps_game.py            Hidden-type RPS: exact Bayes oracle and certificate
  rps_repair.py          RPS CGR, matched random controls and DAgger baselines
  rps_verify.py          Independent RPS certificate verifier
  rps_figure.py          RPS certificate and per-type figure
  memoryless.py          MLP control (9–256–256–9) on soft oracle targets
  eval_memoryless.py     Held-out teacher evaluation for the MLP control
  make_figures.py        Paper figures
tests/                   Fast CPU tests (board legality, ties, symmetries)
data/                    Frozen JSON evidence
paper/                   paper_v2.tex/.pdf and figures
docs/                    Data card and reproduction guide
```

`*.pth`, `runs/`, `checkpoints/`, `artifacts/`, `ablation_logs/` and `.tools/` are
not versioned.

## Scope and limitations

Tic-tac-toe is small enough that “certified optimal” has a precise meaning, which
is why the twin design is possible; the claim is about *certification under
tied optimal teachers*, not about RL benchmarks at scale. The paper reports exact
tie statistics, nine checkpoints (two of which, C-43 and C-44, are distinct
trainings that coincide on every evaluated path), 30-teacher ensembles, an
exact-oracle control, an exact worst-case certificate, model-size controls, a
certificate-guided repair algorithm with matched random controls, and explicit
retractions.
Larger games with tie-rich value functions (Connect Four, Hex, Go endgames) are
the natural next test.

## License and citation

Apache 2.0 (see [LICENSE](LICENSE)); citation metadata in [CITATION.cff](CITATION.cff).

Starlyn Eliezer Rosario Reyes · Independent Researcher · Santo Domingo,
Dominican Republic · starlynelizerrosario@gmail.com
