# Fragility paper v2 — packaging and reproducibility tasks.
# POSIX make; works in Git Bash, Linux and macOS.

PYTHON ?= python
CONFIG ?= B
GAMES ?= 200
RUNS ?= b_s42=ablation_logs
TWIN_SEEDS ?= 1234,5678,9012
OUT ?= data/tbr_eval.json
THREADS ?= 4

.PHONY: install test smoke verify eval-tbr certify repair c4 rps paper clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests -q

# Import/CLI sanity checks plus the fast CPU test suite. Does not train.
smoke:
	PYTHONPATH=src $(PYTHON) -c "import liquid_rl_trainer, ablation_ttt, tbr_eval, human_like_agent, worst_case_eval, distill_history, repair, repair_figure, c4_game, c4_repair, c4_verify, c4_figure, c4_endgame, c4_endgame_verify, rps_game, rps_repair, rps_verify, rps_figure"
	PYTHONPATH=src $(PYTHON) src/tbr_eval.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/ablation_ttt.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/worst_case_eval.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/distill_history.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/repair.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/c4_repair.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/c4_verify.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/c4_endgame.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/rps_repair.py --help >/dev/null
	PYTHONPATH=src $(PYTHON) src/rps_verify.py --help >/dev/null
	$(PYTHON) -m pytest tests -q

# Evaluate every checkpoint dir named in RUNS against M6 fixed, twin mentors,
# a uniform RandomAgent and a weak minimax. Override with e.g.
#   make eval-tbr RUNS="tbr_s42=runs/tbr_B_s42" OUT=data/tbr_eval_B_s42.json
eval-tbr:
	$(PYTHON) src/tbr_eval.py --runs "$(RUNS)" --config $(CONFIG) \
		--games $(GAMES) --twin-seeds $(TWIN_SEEDS) --out $(OUT)

# Exact worst-case certificate over every optimal teacher. Needs the trained
# checkpoints under runs/ and artifacts/; takes ~3 min on CPU.
certify:
	$(PYTHON) src/worst_case_eval.py --preset all \
		--mlp "MLP-s0=runs/memoryless_s0/mlp.pt;MLP-s1=runs/memoryless_s1/mlp.pt" \
		--out data/worst_case.json

# Independent re-certification of every saved artifact, using the verifiers
# that do not share code with the repair scripts. Needs the checkpoints under
# runs/ produced by `make repair`, `make c4` and `make rps` (~10 min on CPU).
verify:
	$(PYTHON) src/worst_case_eval.py --runs "C-42-repaired=C:runs/repair_C42_s0" \
		--out data/worst_case_repair_C42.json
	$(PYTHON) src/c4_verify.py --ckpt runs/c4_canon_s0/c4_gru.pt --hidden 64 \
		--games 2000 --out data/c4_verify_s0.json
	$(PYTHON) src/c4_endgame_verify.py --ckpt runs/c4_end_ctrl_s0/c4_gru.pt \
		--roots data/c4_endgame_roots_8_40_s0.json --out data/c4_end_verify_s0.json
	$(PYTHON) src/rps_verify.py --data data/rps_s0.json --ckpt runs/rps_s0/rps_gru.pt

# Certificate-guided repair on the C-42 agent (~6 min on CPU), followed by an
# independent re-certification of the repaired checkpoint and the figure.
repair:
	$(PYTHON) src/repair.py --config C --ckpt runs/fixed_C_s42/ablation_C.pth \
		--label C42 --eval-all --ckpt-out runs/repair_C42_s0 --out data/repair_C42_s0.json
	$(PYTHON) src/worst_case_eval.py \
		--runs "C-42-repaired=C:runs/repair_C42_s0" \
		--out data/worst_case_repair_C42.json
	$(PYTHON) src/repair_figure.py \
		--runs "C-42 (GRU 300)=data/repair_C42_s0.json;Full-history GRU=data/repair_histfull_s0.json;Broken GRU (20 ep)=data/repair_hist20_s0.json" \
		--cols 3 --panel-width 2.45 --out figs_v2/fig6_repair.pdf

# Connect Three (4x3): exact certificate, CGR, matched controls, and the
# independent verifier, for three seeds. Takes about three minutes on CPU.
c4:
	for s in 0 1 2; do \
		$(PYTHON) src/c4_repair.py --rows 4 --cols 3 --win 3 --hidden 64 \
			--epochs 300 --lr 2e-3 --max-iters 10 --seed $$s \
			--control-seeds 1,2,3 --control-mode all,tree \
			--ckpt-out runs/c4_canon_s$$s --out data/c4_canon_s$$s.json; \
		$(PYTHON) src/c4_verify.py --ckpt runs/c4_canon_s$$s/c4_gru.pt \
			--hidden 64 --games 2000 --out data/c4_verify_s$$s.json; \
	done
	$(PYTHON) src/c4_figure.py --out figs_v2/fig7_c4.pdf

# Hidden-type Rock-Paper-Scissors: exact Bayes oracle, CGR, matched controls,
# DAgger baselines, independent verifier and figure. Seed 0 runs the full
# baseline matrix; seeds 1-2 replicate the CGR arm. Takes about 90 min on CPU.
rps:
	$(PYTHON) src/rps_repair.py --seed 0 --max-iters 12 --threads $(THREADS) \
		--ckpt-out runs/rps_s0 --out data/rps_s0.json
	for s in 1 2; do \
		$(PYTHON) src/rps_repair.py --seed $$s --max-iters 12 --threads $(THREADS) \
			--no-controls --no-dagger \
			--ckpt-out runs/rps_s$$s --out data/rps_s$$s.json; \
		$(PYTHON) src/rps_verify.py --data data/rps_s$$s.json \
			--ckpt runs/rps_s$$s/rps_gru.pt; \
	done
	$(PYTHON) src/rps_verify.py --data data/rps_s0.json --ckpt runs/rps_s0/rps_gru.pt
	$(PYTHON) src/rps_figure.py --seeds 0,1,2 --out figs_v2/fig9_rps.pdf

# Paper build uses a local Tectonic binary (tectonic.exe on Windows). It
# writes paper/paper_v2.pdf next to the source.
PAPER_SRC ?= paper/paper_v2.tex
TECTONIC ?= $(firstword $(wildcard .tools/tectonic.exe .tools/tectonic) tectonic)

paper:
	@if command -v $(TECTONIC) >/dev/null 2>&1; then \
		echo "$(TECTONIC) $(PAPER_SRC)"; \
		$(TECTONIC) $(PAPER_SRC); \
	else \
		echo "tectonic not found (looked for .tools/tectonic.exe, .tools/tectonic, PATH)."; \
		echo "Place the tectonic executable at .tools/tectonic.exe, then run:"; \
		echo "  .tools/tectonic.exe $(PAPER_SRC)"; \
	fi

# Remove build/test caches and LaTeX intermediates. Never touches data, runs,
# checkpoints, paper sources or the paper figures.
clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info src/*.egg-info
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.synctex.gz
