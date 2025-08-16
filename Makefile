# Makefile for Real-Time VAD Library

.PHONY: help install install-dev test test-fast test-slow clean lint format type-check build publish docs dev-setup

# Default target
help:
	@echo "Real-Time VAD Library - Available Commands"
	@echo "=========================================="
	@echo "Development:"
	@echo "  dev-setup     - Set up development environment (run once)"
	@echo "  install       - Install package in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run fast tests only (exclude slow/integration)"
	@echo "  test-slow     - Run slow tests only"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-cov      - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          - Run linting (flake8)"
	@echo "  format        - Format code (black + isort)"
	@echo "  type-check    - Run type checking (mypy)"
	@echo "  check-all     - Run all code quality checks"
	@echo ""
	@echo "Build & Distribution:"
	@echo "  build         - Build package"
	@echo "  publish       - Publish to PyPI"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          - Generate documentation"
	@echo ""
	@echo "Examples:"
	@echo "  example-basic - Run basic usage example"
	@echo "  example-advanced - Run advanced usage example"

# Development setup
dev-setup:
	python setup_dev.py

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,audio,examples]"

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-slow:
	pytest tests/ -v -m "slow"

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src tests examples

format:
	black src tests examples setup_dev.py
	isort src tests examples setup_dev.py

type-check:
	mypy src

check-all: lint type-check
	@echo "All code quality checks completed"

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Documentation
docs:
	@echo "Documentation generation not yet implemented"
	@echo "For now, see README.md"

# Examples
example-basic:
	python examples/basic_usage.py

example-advanced:
	python examples/advanced_usage.py

# Development helpers
install-pre-commit:
	pre-commit install

run-pre-commit:
	pre-commit run --all-files

# Virtual environment helpers (Windows)
venv-windows:
	python -m venv venv
	venv\\Scripts\\activate && pip install --upgrade pip
	venv\\Scripts\\activate && pip install -e ".[dev,audio,examples]"

# Virtual environment helpers (Unix/Linux/macOS)
venv-unix:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -e ".[dev,audio,examples]"

# Quick development workflow
dev: format lint type-check test-fast
	@echo "Development workflow completed successfully!"

# CI/CD workflow
ci: check-all test
	@echo "CI workflow completed successfully!"
