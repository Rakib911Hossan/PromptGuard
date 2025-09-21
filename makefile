# Basic Makefile for GanitLLM Project

# Variables
PYTHON = python3
PIP = pip3
PROJECT_NAME = GanitLLM
VENV_NAME = .venv
PYTHON_VENV = $(VENV_NAME)/bin/python
PIP_VENV = $(VENV_NAME)/bin/pip

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@echo "=================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup targets
.PHONY: setup
setup: ## Set up the development environment
	@echo "Setting up development environment..."
	uv sync

.PHONY: format
format: ## Format code
	@echo "Formatting code..."
	uvx ruff format

# Cleanup targets
.PHONY: clean
clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true

.PHONY: clean-venv
clean-venv: ## Remove virtual environment
	@echo "Removing virtual environment..."
	rm -rf $(VENV_NAME)

.PHONY: clean-all
clean-all: clean clean-venv ## Clean everything including virtual environment

.PHONY: few-shot
few-shot:
	@echo "Running few-shot"
	PYTHONPATH=. uv run src/few_shot.py --debug