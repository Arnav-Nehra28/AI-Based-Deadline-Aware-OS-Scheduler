# RL Training

This module contains the runnable PPO training and evaluation pipeline for the scheduler environment.

## Why This Is an RL Problem

For the RL track, the dataset is not used like a normal labeled classification or regression dataset.

- There is no fixed correct action for every task independent of context.
- The best scheduling decision depends on the current queue, machine capacities, running jobs, and time.
- The policy is learned from reward, not by matching a ground-truth label.

Important nuance:

- The trace still contains `historical_machine_id`.
- In the current environment, that field is context from the trace plus a small reward bonus for matching history.
- It is not the target label that the RL policy tries to copy step by step.

So the RL formulation is:

`state -> action -> reward -> next state`

or, more formally:

`S_t -> A_t -> R_t -> S_{t+1}`

## RL Formulation in This Project

### State

The environment observation is split into three parts:

- `task_features` with normalized task demand, duration, wait time, arrival position, queue size, and resource pressure
- `candidate_features` with a fixed shortlist of machine candidates and their feasibility / fit signals
- `fleet_summary` with queue pressure, running-job count, feasible fraction, and utilization summaries

Current tensor shapes in the environment:

- `task_features`: 12 values
- `candidate_features`: `top_k_candidates x 13`
- `fleet_summary`: 18 values

In simple project terms, the state represents:

- current task
- current machine capacities
- pending queue
- jobs already running
- current scheduler time

### Action

At each step the agent chooses one discrete action:

- assign the task to one of the shortlisted machines
- or choose the special `defer` action

The shortlist is action-masked, so invalid machine slots are hidden from the policy.

### Reward

The agent does not learn from labels. It learns by maximizing the reward signal produced by the environment.

The current reward is the sum of shaped components such as:

- feasible assignment bonus
- wait-time penalty
- balance bonus
- fragmentation penalty
- hotspot penalty
- historical-match bonus
- overload penalty
- invalid-action penalty
- defer penalty

This means the policy is optimized toward better scheduling behavior, not toward reproducing a labeled answer.

### Policy / Output

Conceptually the policy learns:

`pi(a | s)`

meaning a probability distribution over available actions for the current state.

In simple terms:

- input: current task plus system state
- output: choose the best shortlisted machine, or defer

At inference time, `MaskablePPO` returns an action index. The environment then maps that index to:

- a concrete `machine_id`
- or the defer action

So the user-facing final output of the RL model is a scheduling decision.

## What the Dataset Looks Like

The RL environment dataset is organized as:

- `tasks`
- `machines`
- `episodes`
- `metadata`

The core task fields are:

- `task_id`
- `arrival_time`
- `duration`
- `cpu_demand`
- `mem_demand`
- `disk_demand`
- `historical_machine_id`

The machine catalog contains:

- `machine_id`
- `cpu_capacity`
- `mem_capacity`
- `disk_capacity`

This is enough to build an online scheduling simulator, even though it is not a labeled supervised dataset.

## Evaluation

### Metrics Already Produced by the Current RL Code

The current training and evaluation scripts already report RL-centric metrics such as:

- mean total reward
- feasible decision rate
- defer rate
- invalid decision rate
- per-component reward summaries during training

These are useful for checking whether the policy is learning stable decision behavior.

### Final Project Metrics to Report

For the project comparison, the more interpretable scheduling metrics should be:

1. Deadline Miss Ratio (DMR)

`DMR = missed_deadline_tasks / total_tasks`

2. Waiting Time

`waiting_time = start_time - arrival_time`

3. Turnaround Time (TAT)

`TAT = completion_time - arrival_time`

4. CPU Utilization

`CPU_utilization = CPU_busy_time / total_time`

Recommended comparison table:

| Scheduler | Deadline Miss Ratio | Waiting Time | Turnaround Time | CPU Utilization |
| --- | --- | --- | --- | --- |
| FCFS |  |  |  |  |
| SJF |  |  |  |  |
| RL Model |  |  |  |  |

### Deadline Rule Used by the New Evaluation Flow

The RL dataset does not contain a native deadline column, so the project evaluation scripts use:

`deadline = arrival_time + slack_factor * duration`

You can control `slack_factor` with:

- `--deadline-slack-factor` in validation, test, and final comparison commands

This keeps DMR reproducible and comparable across FCFS, SJF, and RL.

## Practical Workflow

### Prerequisites

Use a repo-local pip cache so interrupted installs can reuse downloads:

```bash
mkdir -p .pip-cache
export PIP_CACHE_DIR="$(pwd)/.pip-cache"
```

This project should be treated as a GPU-backed training workflow.

Before training, verify that the GPU stack is available:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Expected result:

- `nvidia-smi` should list the NVIDIA device
- `torch.cuda.is_available()` should print `True`

#### GPU Install (Recommended)

Install a CUDA-enabled PyTorch build first, then the RL libraries:

```bash
mkdir -p .pip-cache
export PIP_CACHE_DIR="$(pwd)/.pip-cache"
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium stable-baselines3 sb3-contrib
```

If you also plan to run the supervised XGBoost baseline and plotting code in the same environment, keep these available too:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib xgboost
```

Optional but helpful for plotting on shared or headless systems:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

#### CPU Fallback

If you need a non-GPU fallback for debugging only:

```bash
pip install gymnasium stable-baselines3 sb3-contrib
```

## Train MaskablePPO

```bash
python -m training.rl.train_ppo \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --total-timesteps 200000 \
  --learning-rate 3e-4 \
  --gamma 0.99 \
  --batch-size 64 \
  --n-epochs 5 \
  --clip-range 0.2 \
  --ent-coef 0.01
```

Attention-based extractor (recommended for better candidate reasoning):

```bash
python -m training.rl.train_ppo \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --run-name stress_attention_v1 \
  --features-extractor attention \
  --features-dim 256 \
  --attention-heads 4 \
  --total-timesteps 500000 \
  --machine-capacity-scale 0.35 \
  --machine-pool-size 32 \
  --max-steps 500 \
  --n-envs 8 \
  --n-steps 2048 \
  --batch-size 512 \
  --n-epochs 10 \
  --gamma 0.997 \
  --gae-lambda 0.98 \
  --learning-rate 1e-4 \
  --learning-rate-schedule warmup_cosine \
  --lr-warmup-steps 50000 \
  --lr-min 1e-5
```

Artifacts are written under `artifacts/rl/<run_name>/`:

- `final_model.zip`
- `run_config.json`
- `train_summary.json`
- `checkpoints/`
- `eval/` if eval callback is enabled

## Evaluate a Saved Policy

```bash
python -m training.rl.evaluate_policy \
  --model-path artifacts/rl/<run_name>/final_model.zip \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --episodes 20 \
  --output-json artifacts/rl/eval/latest_eval.json
```

Evaluate with hybrid fallback policy:

```bash
python -m training.rl.evaluate_model \
  --run-dir artifacts/rl/<run_name> \
  --device cuda \
  --include-hybrid-rl \
  --hybrid-defer-wait-ratio-threshold 2.0 \
  --hybrid-high-utilization-threshold 0.90
```

Run multi-seed evaluation (mean/std stability):

```bash
python -m training.rl.evaluate_multiseed \
  --run-dir artifacts/rl/<run_name> \
  --device cuda \
  --seeds 13,23,37,42,77
```

## Advanced Training Entrypoints

Curriculum training (easy -> medium -> stress):

```bash
python -m training.rl.train_curriculum \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --run-name curriculum_v1 \
  --phase-timesteps 100000,100000,200000 \
  --phase-machine-capacity-scales 0.8,0.5,0.35 \
  --phase-machine-pool-sizes 128,96,32 \
  --phase-learning-rates 3e-4,1e-4,5e-5 \
  --phase-ent-coefs 0.03,0.02,0.01
```

Population-based training search:

```bash
python -m training.rl.train_pbt \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --population-size 6 \
  --rounds 3 \
  --timesteps-per-round 120000
```

Recurrent PPO baseline (LSTM policy):

```bash
python -m training.rl.train_recurrent_ppo \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --total-timesteps 300000 \
  --machine-capacity-scale 0.35 \
  --machine-pool-size 32
```

## Train / Validate / Test / Evaluate Files

The complete experiment workflow now has separate entry points:

- `training/rl/train_model.py`
- `training/rl/validate_model.py`
- `training/rl/test_model.py`
- `training/rl/evaluate_model.py`

### 1. Train (with train/val/test splits)

```bash
python -m training.rl.train_model \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --machine-capacity-scale 1.0 \
  --machine-pool-size 128 \
  --total-timesteps 200000
```

This creates:

- trained PPO artifacts under `artifacts/rl/<run_dir>/`
- `episode_splits.json` with deterministic train/val/test episode ids

### 2. Validate

```bash
python -m training.rl.validate_model \
  --run-dir artifacts/rl/<run_dir> \
  --device cuda \
  --machine-capacity-scale 1.0 \
  --machine-pool-size 128 \
  --deadline-slack-factor 2.0
```

Output:

- `artifacts/rl/<run_dir>/validation_metrics.json`

### 3. Test

```bash
python -m training.rl.test_model \
  --run-dir artifacts/rl/<run_dir> \
  --device cuda \
  --machine-capacity-scale 1.0 \
  --machine-pool-size 128 \
  --deadline-slack-factor 2.0
```

Output:

- `artifacts/rl/<run_dir>/test_metrics.json`

### 4. Final Evaluation (FCFS vs SJF vs RL)

```bash
python -m training.rl.evaluate_model \
  --run-dir artifacts/rl/<run_dir> \
  --device cuda \
  --machine-capacity-scale 1.0 \
  --machine-pool-size 128 \
  --deadline-slack-factor 2.0
```

Outputs:

- `artifacts/rl/<run_dir>/evaluation/comparison_metrics.json`
- `artifacts/rl/<run_dir>/evaluation/comparison_table.csv`
- `artifacts/rl/<run_dir>/evaluation/comparison_table.md`

This command is the one that produces the final project comparison table:

| Scheduler | Deadline Miss Ratio | Waiting Time | Turnaround Time | CPU Utilization |
| --- | --- | --- | --- | --- |
| FCFS | ... | ... | ... | ... |
| SJF | ... | ... | ... | ... |
| RL Model | ... | ... | ... | ... |

## Improving RL Relative to FCFS/SJF

If all schedulers tie, the workload is usually too easy. You can make the benchmark more discriminative by reducing effective machine capacity consistently across train/val/test/evaluate:

```bash
python -m training.rl.train_model \
  --dataset data/interim/rl_env_dataset.json.gz \
  --device cuda \
  --machine-capacity-scale 0.35 \
  --machine-pool-size 64 \
  --total-timesteps 200000
```

Then run validate/test/evaluate with the same `--machine-capacity-scale` and `--machine-pool-size` values.

Guideline:

- `1.0`: original capacity (often easy)
- `0.5`: moderate contention
- `0.35` to `0.25`: hard contention where policy quality differences are easier to observe
- `machine-pool-size`: reduce available machines (for example `64`) to increase contention and force better scheduling

## Recommended Next Steps

To turn the current RL pipeline into a full project comparison setup:

1. Run `train_model.py` to generate a run directory and split file.
2. Run `validate_model.py` and tune hyperparameters if needed.
3. Run `test_model.py` once validation behavior is acceptable.
4. Run `evaluate_model.py` for the final FCFS/SJF/RL comparison table.
5. Add plots from the generated JSON/CSV artifacts into your report.
