# Contributing to the AI-Based Deadline-Aware OS Scheduler

First off, thank you for considering contributing to this project! It's people like you that make open source such a great community.

## Development Workflow

### 1. Setup the Environment
We recommend using a virtual environment. Once activated, use the provided `Makefile` to install dependencies:
```bash
make setup
```

### 2. Project Structure
To help you navigate the codebase, here is a high-level overview of our architecture:

- `rl_pipeline/`: Core reinforcement learning environment (`environment.py`) and datasets. This is where the `TaskSchedulingEnv` resides, along with custom observation spaces.
- `training/`: Contains the logic for training the `MaskablePPO` agent (`train_agent.py`) and custom feature extractors (`attention_extractor.py`).
- `data_preprocessing/`: Scripts for converting raw task/machine traces into the structured format required by the RL environment.
- `tests/`: Comprehensive unit tests ensuring environment stability, reward bounds, and action masking correctness.
- `docs/`: Result charts and documentation imagery.
- `academic_archive/`: Historical papers, reports, and scratch scripts for generating figures used during the project's original academic defense.

### 3. Running Tests
Before submitting any changes, ensure that all tests pass. We rely on tests to guarantee that environment modifications (e.g., action masks, time evolution, reward signals) do not break the reinforcement learning contract.

Run the test suite via the Makefile:
```bash
make test
```

### 4. Running Validation
We also have a robust behavioral validation script to sanity check the RL environment logic.
```bash
make eval
```

### 5. Code Style
Please adhere to PEP 8 standards. We recommend using `black` for formatting and `flake8` for linting before submitting a PR.

## Submitting Pull Requests
- Create a new branch for your feature or bugfix (`git checkout -b feature/my-feature`).
- Ensure all tests pass.
- Submit a PR with a clear description of the changes, the rationale behind them, and any related issue numbers.

We look forward to reviewing your contributions!
