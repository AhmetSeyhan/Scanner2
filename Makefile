.PHONY: install install-dev test lint run run-dashboard docker-build docker-up clean

# ── Install ──────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pip install pydantic-settings

install-dev: install
	pip install -e ".[dev]"

# ── Quality ──────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=core --cov=utils --cov=services --cov=detectors --cov=config --cov=pentashield \
		--cov-report=term-missing --cov-report=html:htmlcov

lint:
	ruff check core/ utils/ services/ detectors/ config/ pentashield/ preprocessing/ api.py api_v1.py

# ── Run ──────────────────────────────────────────────────
run:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard.py

# ── Docker ───────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── PentaShield ─────────────────────────────────────────
test-pentashield:
	pytest tests/ -v -k pentashield

# ── Cleanup ──────────────────────────────────────────────
clean:
	rm -rf htmlcov coverage.xml .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
