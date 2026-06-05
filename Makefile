.PHONY: help install dev-install test lint format clean run docs

help:
	@echo "Available commands:"
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  run          - Run training with default settings"
	@echo "  docs         - View documentation"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

run:
	./run_fed_evo_rl.sh

docs:
	@echo "Documentation available at: docs/README.md"
	@cat docs/README.md | head -50
