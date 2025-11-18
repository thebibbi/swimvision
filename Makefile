.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down pre-commit

help:
	@echo "SwimVision Pro - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo "  make pre-commit      Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run all tests with coverage"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make lint            Run linters (ruff, mypy)"
	@echo "  make format          Format code with ruff"
	@echo "  make clean           Clean build artifacts and cache"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker images"
	@echo "  make docker-up       Start development environment"
	@echo "  make docker-down     Stop development environment"
	@echo "  make docker-logs     View container logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate      Create database migration"
	@echo "  make db-upgrade      Apply database migrations"
	@echo "  make db-shell        Open PostgreSQL shell"

# ========================================
# Setup
# ========================================
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

pre-commit:
	pre-commit install
	pre-commit run --all-files

# ========================================
# Testing
# ========================================
test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

test-unit:
	pytest tests/unit/ -v --cov=src --cov-report=term

test-integration:
	pytest tests/integration/ -v

test-watch:
	ptw -- tests/ -v

# ========================================
# Code Quality
# ========================================
lint:
	@echo "Running Ruff linter..."
	ruff check src/ tests/
	@echo "Running MyPy type checker..."
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code with Ruff..."
	ruff check --fix src/ tests/
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

# ========================================
# Cleanup
# ========================================
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage

# ========================================
# Docker
# ========================================
docker-build:
	docker-compose build

docker-up:
	docker-compose --profile dev up -d
	@echo "SwimVision is running at http://localhost:8501"

docker-up-prod:
	docker-compose --profile prod up -d

docker-up-gpu:
	docker-compose --profile gpu up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f app-dev

docker-shell:
	docker-compose exec app-dev bash

docker-clean:
	docker-compose down -v
	docker system prune -f

# ========================================
# Database
# ========================================
db-migrate:
	alembic revision --autogenerate -m "$(message)"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-shell:
	docker-compose exec postgres psql -U swimvision -d swimvision

# ========================================
# Run Application
# ========================================
run:
	streamlit run app.py

run-debug:
	streamlit run app.py --logger.level=debug

# ========================================
# Development Utilities
# ========================================
download-models:
	python scripts/download_models.py

setup-dev: install-dev pre-commit
	@echo "Development environment ready!"
