# RL-Based Task Scheduler

This repository contains a reinforcement learning pipeline for cloud task scheduling, along with the preprocessing and evaluation code used to compare the learned scheduler against standard heuristic baselines such as `FCFS`, `SJF`, and `Round Robin`.

The project is organized so that the repository tracks source code, tests, and lightweight documentation, while generated datasets, trained models, reports, and experiment artifacts stay out of Git.

## Project Overview

The main idea is to treat scheduling as a sequential decision problem:

`state -> action -> reward -> next_state`

In this project:

- the **state** includes task features, queue pressure, machine capacity, and running-job context
- the **action** is selecting a machine for a task or deferring the decision
- the **reward** encourages feasible assignments, lower deadline misses, better throughput, and stronger scheduling quality

This repository includes:

- preprocessing code to turn raw trace data into model-ready and RL-ready datasets
- a Gym-compatible scheduler environment
- PPO / MaskablePPO training scripts
- validation, testing, and baseline comparison scripts
- report generation utilities for metrics, charts, and PDF summaries

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── data_preprocessing/
├── rl_pipeline/
├── training/
│   └── rl/
├── tests/
└── walkthrough.md
```

Important folders:

- `data_preprocessing/`: raw-to-processed and RL dataset preparation scripts
- `rl_pipeline/`: scheduler environment and dataset helpers
- `training/rl/`: RL training, evaluation, reporting, and experiment split logic
- `tests/`: environment and evaluation checks

## What Is Tracked

This repository is intended to track:

- source code
- test code
- lightweight project documentation
- reproducible scripts for training and evaluation

This repository intentionally does **not** track:

- `artifacts/`
- `results/`
- `venv/`
- generated PDFs
- trained model weights such as `.zip` checkpoints
- prepared datasets and local raw/processed data outputs

Those files are ignored through `.gitignore`.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If you are using GPU-backed PyTorch, install the CUDA-enabled PyTorch build that matches your system before running RL training. Example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then verify GPU availability:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start

### 1. Build the preprocessing outputs

```bash
python -m data_preprocessing
```

### 2. Train an RL scheduler

```bash
python -m training.rl.train_model \
  --run-name my_scheduler_run \
  --device cuda \
  --total-timesteps 200000
```

### 3. Evaluate the trained model

```bash
python -m training.rl.evaluate_model \
  --run-dir artifacts/rl/my_scheduler_run \
  --device cuda
```

### 4. Generate a consolidated metrics report

```bash
python -m training.rl.generate_performance_metrics_file \
  --run-dir artifacts/rl/my_scheduler_run
```

## Current Evaluation Focus

The project evaluates RL against:

- `FCFS`
- `SJF`
- `RR`

Core metrics include:

- Deadline Miss Ratio
- Mean Waiting Time
- Mean Turnaround Time
- CPU Utilization
- Assignment Rate

## Additional Documentation

- `training/rl/README.md`
- `data_preprocessing/README.md`
- `walkthrough.md`

## Authors

- Arnav Nehra
- Rahul Thakur
- Uday Thakur

## License

This project is released under the MIT License. See `LICENSE`.
