# The Fragility of Optimal-Agent Training

**Continuous-Time Dynamics and Adaptive Computation as Heuristic Regularizers in Edge AI**

This repository contains the complete experimental code and data for the paper investigating how continuous-time neural networks (CfC) and adaptive computation (ACT) behave under adversarial RL training. The central finding is a **Hardware Paradox**: techniques that look optimal on paper introduce structural fragilities that manifest only at deployment on resource-constrained hardware.

## Key Findings

| Finding | Evidence |
|---|---|
| **Non-transitive dynamics** | GRU (C) defeats CfC+ACT (A) 100–0 despite A beating Random 96%. Rock-Paper-Scissors cycle: A→Random→C→A |
| **Hardware Paradox** | ACT triples win rate (3%→9%) at 2.4× latency. On CPU, the overhead negates the theoretical efficiency gain |
| **Negative Transfer** | TTT backbone actively harms Connect Four learning: 23.3% vs 80.0% from scratch (−56.7pp). Peak 70% at game 2,600 then collapse |
| **False-Master Effect** | Agents dominate the opponent they trained against but lose to unfamiliar strategies |
| **Human-like curriculum fails** | REINFORCE + bounded-rational opponents collapses to 2.5% WR vs Random |

## Repository Structure

```
fragility-paper/
├── paper/               # LaTeX source + figures
│   ├── paper.tex
│   ├── fig1_nontransitivity.pdf
│   └── fig2_transfer_curves.pdf
├── src/                 # Experimental code
│   ├── liquid_rl_trainer.py    # Core CfC training loop (REINFORCE + GAE)
│   ├── ablation_ttt.py         # 4-config ablation study
│   ├── tournament.py           # Round-robin tournament
│   ├── evaluate_cross.py       # Cross-evaluation matrix
│   ├── transfer_connect4.py    # TTT→Connect Four transfer
│   ├── benchmark_act.py        # ACT wall-clock benchmark
│   ├── human_like_agent.py     # Bounded-rational minimax
│   ├── connect4_env.py         # Connect Four environment
│   ├── minimax_mentor.py       # Minimax with stochastic noise
│   └── generate_figures.py     # Paper figures from data
├── data/                # Experimental results (JSON)
│   ├── ablation_results.json
│   ├── tournament_results.json
│   ├── cross_eval_results.json
│   ├── transfer_results.json
│   └── benchmark_act_results.json
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib (for figures)

## Reproducing Experiments

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

### Transfer learning
```bash
python src/transfer_connect4.py
```

### Benchmark ACT latency
```bash
python src/benchmark_act.py
```

### Compile the paper
```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

## Architecture Overview

All agents share the same structure:

```
Observation → InputAdapter → CfC/GRU Backbone → PolicyHead + ValueHead
```

- **CfC cell**: Closed-form continuous-time (LNN), hidden_dim=300, state clamped to [−5, 5]
- **ACT**: Adaptive Computation Time with soft-halting accumulation (Graves 2016)
- **GRU**: Discrete-time baseline, same hidden dimension, 1 step per decision
- **Training**: REINFORCE + GAE (λ=0.95, γ=0.99), curriculum with EMA shadow

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{rosario2026fragility,
  title={The Fragility of Optimal-Agent Training:
         Continuous-Time Dynamics and Adaptive Computation
         as Heuristic Regularizers in Edge AI},
  author={Rosario, Starlyn},
  year={2026},
  doi={10.5281/zenodo.21077402}
}
```

## Contact

Starlyn Rosario — Independent Researcher

[Preprint on Zenodo](https://zenodo.org) | [GitHub](https://github.com/starlyn/fragility-paper)
