# Trida AI Blueprint — Plant Operations Loop Agent
#
# Primary workflow is uv (https://docs.astral.sh/uv/):
#   make install     uv sync --extra dev
#   make demo        the seeded shift board, worked end to end, offline
#   make test        pytest
#   make evals       golden loop gate (offline)
#   make gate        reasoner regression gate: stub candidate vs the committed baseline
#   make serve       the API + console on http://localhost:8000
#
# No uv? python3 -m venv .venv && source .venv/bin/activate, then:
#   make install-pip pip install -e ".[dev]" -c constraints.txt

.PHONY: install install-pip demo test evals gate serve

install:
	uv sync --extra dev

install-pip:
	# The frozen set first (hash-verified), then the package itself with
	# --no-deps: pip refuses an editable install under a hashed
	# constraints file ("no single file to hash"), so the two never mix.
	pip install -r constraints.txt
	pip install --no-deps -e .
	pip install "pytest==9.1.1" "httpx==0.28.1"

demo:
	uv run plant-loop-demo

test:
	uv run pytest -q

evals:
	uv run python evals/run_evals.py

gate:
	uv run python evals/run_gate.py check --backend stub

serve:
	uv run uvicorn plant_loop.api:app --port 8000
