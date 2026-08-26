# The Fragility of Optimal-Agent Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21077402.svg)](https://doi.org/10.5281/zenodo.21077402)
[![CI](https://github.com/starlyn2010/fragility-paper/actions/workflows/ci.yml/badge.svg)](https://github.com/starlyn2010/fragility-paper/actions/workflows/ci.yml)

**Continuous-Time Dynamics and Adaptive Computation as Heuristic Regularizers in Edge AI**

> Training lightweight recurrent agents (CfC/LNN and GRU) against a near-perfect Minimax opponent in Tic-Tac-Toe yields policies that draw 100% against the optimal adversary yet collapse against stochastic opponents. This repository contains the complete code, data, and LaTeX source for the systematic study of that brittleness — including a hardware paradox, non-transitive cycles, and negative transfer to Connect Four.

---

## Abstract

We present counterintuitive empirical findings that challenge assumptions in RL for Edge AI. Lightweight agents based on Closed-form Continuous-time Neural Networks (CfC) and GRUs, trained via REINFORCE + GAE against a Minimax mentor with a 3-phase curriculum (Random → Shadow EMA → Shadow+Minimax), achieve optimal play against the teacher but not general competence. An ablation across four configurations (CfC ± ACT, GRU) reveals: (i) ACT triples win-rate (3%→9%) at 2.4× latency — a hardware paradox where theoretical efficiency is negated by wall-clock overhead on CPU; (ii) non-transitive dominance (GRU defeats CfC+ACT 100–0 despite losing to Random); and (iii) negative transfer (TTT pre-training degrades Connect Four from 80% to 23%). We argue that optimality against a single adversary induces geometric over-specialization, and that continuous-time dynamics and adaptive halting function as heuristic regularizers whose benefit is deployment-contingent.

---

## Key Findings

| Finding | Evidence |
|---|---|
| **Non-transitive dynamics** | GRU (C) defeats CfC+ACT (A) 100–0 despite A beating Random 96%. Cycle: A→Random→C→A |
| **Hardware Paradox** | ACT triples win rate (3%→9%) at 2.4× latency. On CPU, overhead negates theoretical gain |
| **Negative Transfer** | TTT backbone harms Connect Four: 23.3% vs 80.0% from scratch (−56.7pp). Peak 70% at game 2,600 then collapse |
| **False-Master Effect** | Agents dominate their training opponent but lose to unfamiliar strategies |
| **Human-like curriculum fails** | REINFORCE + bounded-rational opponents collapses to 2.5% WR vs Random |

All results are scoped to the frozen JSON logs in `data/` — see Reproducibility.

---

## Repository Structure

```
fragility-paper/
├── paper/               # LaTeX source + figures
│   ├── paper.tex
│   ├── fig1_nontransitivity.pdf
│   └── fig2_transfer_curves.pdf
├── src/                 # Experimental code
│   ├── liquid_rl_trainer.py    # Core CfC training loop (REINFORCE + GAE, EMA shadow, 3-phase curriculum)
│   ├── ablation_ttt.py         # 4-config ablation study
│   ├── tournament.py           # Round-robin tournament
│   ├── evaluate_cross.py       # Cross-evaluation matrix
│   ├── transfer_connect4.py    # TTT→Connect Four transfer (probe→finetune)
│   ├── benchmark_act.py        # ACT wall-clock benchmark
│   ├── human_like_agent.py     # Bounded-rational minimax (asymmetric depth + Gaussian noise)
│   ├── connect4_env.py         # Connect Four environment (7×6, 4-in-a-row)
│   ├── minimax_mentor.py       # Minimax with stochastic noise
│   └── generate_figures.py     # Paper figures from data/
├── data/                # Experimental results (JSON, frozen)
│   ├── ablation_results.json
│   ├── tournament_results.json
│   ├── cross_eval_results.json
│   ├── transfer_results.json
│   └── benchmark_act_results.json
├── tests/test_import.py # smoke tests
├── pyproject.toml / requirements.txt
├── CITATION.cff
├── LICENSE (Apache 2.0)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/starlyn2010/fragility-paper.git
cd fragility-paper
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # torch, numpy, matplotlib + pytest/ruff
# or minimal:
pip install -r requirements.txt
```

**Tested:** Python 3.10–3.12, `torch 2.x+cpu`, `numpy`, `matplotlib` on Linux CPU. No GPU required; all runs are CPU-friendly.

Verify setup:

```bash
pytest -q
python src/generate_figures.py  # regenerates paper/fig*.pdf from data/
```

---

## Usage

### Quick sanity check (minutes)

```bash
python src/ablation_ttt.py --quick          # 2 configs × 1000 games
python src/generate_figures.py              # generate paper figures
```

### Full ablation (≈3 hours on CPU)

```bash
python src/ablation_ttt.py                  # 4 configs × 5000 games
```

### Tournament

```bash
python src/tournament.py                    # round-robin, 200 games/pair
```

### Cross-evaluation matrix

```bash
python src/evaluate_cross.py
```

### Transfer learning (TTT → Connect Four)

```bash
python src/transfer_connect4.py             # full: probe + finetune + scratch baseline
python src/transfer_connect4.py --quick     # 500+1000 games
```

### Benchmark ACT latency

```bash
python src/benchmark_act.py
```

### Compile the paper

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
# or
pdflatex paper.tex; bibtex paper; pdflatex paper.tex; pdflatex paper.tex
```

---

## Results

Results below are taken verbatim from the frozen logs in `data/` (no re-running needed). See `paper/paper.tex` Table 1–3 and Figures 1–2.

### Ablation (`data/ablation_results.json`, 4 configs × 5000 games)

* **CfC + ACT:** WR ~9%, steps/decision ~3.1 (adaptive)
* **CfC:** WR ~3%, steps/decision 1.0
* **GRU:** intermediate; **GRU vs CfC+ACT head-to-head:** 100–0 (deterministic)

### Tournament (`data/tournament_results.json`, 200 games/pair)

Non-transitive cycle confirmed: A (CfC+ACT) beats Random 96%; Random beats C (GRU) 89%; C beats A 100–0.

### Transfer (`data/transfer_results.json`, TTT→C4)

* From scratch: **80.0%** vs Random (C4)
* From TTT backbone: **23.3%** (peak 70% at game 2,600 then collapse to 23% — catastrophic forgetting / latent fragmentation)

### Hardware Paradox (`data/benchmark_act_results.json`)

ACT: 2.45× wall-clock training time for +6pp win-rate. On resource-constrained CPU, the overhead dominates.

> Honesty note: all claims are limited to the Tic-Tac-Toe / Connect Four setup with the stated hidden dim (300), clamp [−5,5], and REINFORCE+GAE. No claim of universality to other games or optimizers.

---

## Reproducibility

* **Seeds:** fixed per run; EMA shadow (`polyak`) and curriculum phase boundaries are logged in `liquid_rl_trainer.py:TrainingConfig`.
* **Determinism:** Python `random`, `numpy`, `torch` seeded; `Categorical` sampling uses torch RNG.
* **Checkpoint gate:** saves only if WR ∈ [0.45, 0.75], ‖g‖ < 10.0 and entropy > 0.1 — prevents degenerate saves.
* **Hardware:** all logs measured on CPU (no GPU required). Wall-clock numbers are machine-dependent; relative ratios are stable.
* **Data:** `data/*.json` are committed and used by `generate_figures.py` to reproduce paper figures without re-running RL.

---

## Citation

```bibtex
@article{rosario2026fragility,
  title  = {The Fragility of Optimal-Agent Training:
            Continuous-Time Dynamics and Adaptive Computation
            as Heuristic Regularizers in Edge AI},
  author = {Rosario, Starlyn},
  year   = {2026},
  doi    = {10.5281/zenodo.21077402},
  url    = {https://github.com/starlyn2010/fragility-paper}
}
```

Also see `CITATION.cff` and [Zenodo DOI 10.5281/zenodo.21077402](https://doi.org/10.5281/zenodo.21077402).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgments

* Hasani et al. (CfC / Liquid Time-constant Networks), Graves (ACT), and the Minimax literature for opponents.
* The reviewers and open-source PyTorch / NumPy / Matplotlib communities.

---

## Contact

**Starlyn Rosario** — Independent Researcher, Santo Domingo, Dominican Republic — `starlyneliezerrosario@gmail.com`

[Preprint on Zenodo](https://doi.org/10.5281/zenodo.21077402) · [GitHub](https://github.com/starlyn2010/fragility-paper)
