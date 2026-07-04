.PHONY: help setup test train eval lint format clean

# Default python command
PYTHON = python3

help:
	@echo "Available commands:"
	@echo "  make setup    - Install project dependencies"
	@echo "  make test     - Run all unit tests"
	@echo "  make train    - Run the reinforcement learning training pipeline"
	@echo "  make eval     - Validate the environment and dataset scale"
	@echo "  make clean    - Remove cached Python files and virtual environment artifacts"

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover tests

train:
	@echo "Starting RL training pipeline..."
	$(PYTHON) training/rl/train_agent.py

eval:
	@echo "Running environment validation..."
	$(PYTHON) -m rl_pipeline.validate_environment

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
