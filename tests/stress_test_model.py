"""
Comprehensive Stress Test for the AI-Based Deadline-Aware OS Scheduler.

This script generates synthetic stress data at scale and evaluates:
  1. XGBoost Classifier (machine assignment) — accuracy under adversarial conditions
  2. XGBoost Regressor (duration prediction) — error metrics under distribution shift
  3. RL Environment (TaskSchedulingEnv) — scheduling performance under resource pressure

Stress scenarios tested:
  A. High-volume: 10x the normal task count
  B. Extreme resource pressure: demand near/exceeding machine capacity
  C. Distribution shift: unseen feature ranges (out-of-distribution)
  D. Edge cases: zero-duration tasks, identical tasks, single-machine bottleneck
  E. RL stress: tight capacity, many tasks, few machines
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULT_PATH = PROJECT_ROOT / "results" / "stress_test_results"
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# PART 1: XGBoost Model Stress Tests
# ============================================================

def load_xgboost_models():
    """Load the trained XGBoost classifier and regressor."""
    import joblib
    model_dir = PROJECT_ROOT / "results" / "XGBoost_result"
    clf_path = model_dir / "xgb_classifier.pkl"
    reg_path = model_dir / "xgb_regressor.pkl"

    if not clf_path.exists() or not reg_path.exists():
        print("⚠️  XGBoost models not found. Skipping XGBoost stress tests.")
        return None, None

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return clf, reg


def create_features(df):
    """Replicate the feature engineering from the training pipeline."""
    df = df.copy()
    df["cpu_ratio"] = df["cpu_task"] / df["cpu_machine"].replace(0, np.nan)
    df["mem_ratio"] = df["mem_task"] / df["mem_machine"].replace(0, np.nan)
    df["disk_ratio"] = df["disk_task"] / df["disk_machine"].replace(0, np.nan)
    df["cpu_gap"] = df["cpu_machine"] - df["cpu_task"]
    df["mem_gap"] = df["mem_machine"] - df["mem_task"]
    df["resource_pressure"] = df["cpu_ratio"] + df["mem_ratio"] + df["disk_ratio"]
    df["task_hour"] = (df["start_time"] % 86400) // 3600
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


def generate_stress_data_xgb(scenario: str, n_samples: int = 5000) -> pd.DataFrame:
    """Generate stress data for various adversarial scenarios."""
    rng = np.random.default_rng(42)
    machines = [f"m{i}" for i in range(1, 51)]  # 50 machines

    if scenario == "high_volume":
        # Normal distribution but 10x volume
        data = {
            "machine_id": rng.choice(machines, n_samples),
            "cpu_task": rng.uniform(0.01, 0.5, n_samples),
            "mem_task": rng.uniform(0.01, 0.5, n_samples),
            "disk_task": rng.uniform(0.0, 0.3, n_samples),
            "start_time": rng.uniform(0, 86400, n_samples),
            "duration": rng.exponential(50, n_samples),
            "cpu_machine": rng.uniform(0.3, 1.0, n_samples),
            "mem_machine": rng.uniform(0.3, 1.0, n_samples),
            "disk_machine": rng.uniform(0.0, 1.0, n_samples),
        }

    elif scenario == "extreme_resource_pressure":
        # Tasks demand NEAR or EXCEEDING machine capacity
        data = {
            "machine_id": rng.choice(machines, n_samples),
            "cpu_task": rng.uniform(0.7, 1.5, n_samples),    # very high demand
            "mem_task": rng.uniform(0.6, 1.2, n_samples),
            "disk_task": rng.uniform(0.4, 1.0, n_samples),
            "start_time": rng.uniform(0, 86400, n_samples),
            "duration": rng.exponential(100, n_samples),
            "cpu_machine": rng.uniform(0.5, 1.0, n_samples),  # capacity may not match
            "mem_machine": rng.uniform(0.5, 1.0, n_samples),
            "disk_machine": rng.uniform(0.3, 0.8, n_samples),
        }

    elif scenario == "distribution_shift":
        # Entirely different scale — values the model has never seen
        data = {
            "machine_id": rng.choice(machines, n_samples),
            "cpu_task": rng.uniform(2.0, 10.0, n_samples),   # way out of training range
            "mem_task": rng.uniform(2.0, 10.0, n_samples),
            "disk_task": rng.uniform(1.0, 5.0, n_samples),
            "start_time": rng.uniform(86400, 172800, n_samples),
            "duration": rng.uniform(500, 5000, n_samples),
            "cpu_machine": rng.uniform(5.0, 20.0, n_samples),
            "mem_machine": rng.uniform(5.0, 20.0, n_samples),
            "disk_machine": rng.uniform(2.0, 10.0, n_samples),
        }

    elif scenario == "edge_cases":
        # Mix of zero-duration, identical tasks, single machine
        n_half = n_samples // 2
        data = {
            "machine_id": ["m1"] * n_samples,  # all tasks to one machine
            "cpu_task": np.concatenate([np.zeros(n_half), np.full(n_samples - n_half, 0.5)]),
            "mem_task": np.concatenate([np.zeros(n_half), np.full(n_samples - n_half, 0.5)]),
            "disk_task": np.zeros(n_samples),
            "start_time": np.concatenate([np.zeros(n_half), rng.uniform(0, 86400, n_samples - n_half)]),
            "duration": np.concatenate([np.zeros(n_half), np.full(n_samples - n_half, 60.0)]),
            "cpu_machine": np.full(n_samples, 0.5),
            "mem_machine": np.full(n_samples, 0.6),
            "disk_machine": np.zeros(n_samples),
        }

    elif scenario == "noisy_data":
        # Normal data with random NaN injection and extreme outliers
        data = {
            "machine_id": rng.choice(machines, n_samples),
            "cpu_task": rng.uniform(0.01, 0.5, n_samples),
            "mem_task": rng.uniform(0.01, 0.5, n_samples),
            "disk_task": rng.uniform(0.0, 0.3, n_samples),
            "start_time": rng.uniform(0, 86400, n_samples),
            "duration": rng.exponential(50, n_samples),
            "cpu_machine": rng.uniform(0.3, 1.0, n_samples),
            "mem_machine": rng.uniform(0.3, 1.0, n_samples),
            "disk_machine": rng.uniform(0.0, 1.0, n_samples),
        }
        df = pd.DataFrame(data)
        # Inject 10% NaN values
        mask = rng.random(df.shape) < 0.10
        df = df.mask(mask[:, :len(df.columns)])
        df["machine_id"] = rng.choice(machines, n_samples)  # restore machine_id
        return create_features(df)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return create_features(pd.DataFrame(data))


def stress_test_xgboost(clf, reg):
    """Run all XGBoost stress test scenarios."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_absolute_error, mean_squared_error, r2_score
    )
    from sklearn.preprocessing import LabelEncoder

    scenarios = [
        "high_volume",
        "extreme_resource_pressure",
        "distribution_shift",
        "edge_cases",
        "noisy_data",
    ]

    results = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  STRESS SCENARIO: {scenario.upper().replace('_', ' ')}")
        print(f"{'='*60}")

        df = generate_stress_data_xgb(scenario, n_samples=5000)

        feature_cols = [c for c in df.columns if c not in ["machine_id", "duration"]]
        X = df[feature_cols].astype(np.float32)
        y_true_clf = df["machine_id"]
        y_true_reg = df["duration"]

        # --- Classifier evaluation ---
        try:
            start = time.time()
            pred_clf = clf.predict(X)
            inference_time = time.time() - start

            # Decode predictions back to labels
            le = LabelEncoder()
            le.classes_ = clf.classes_ if hasattr(clf, 'classes_') else np.array([])

            # Since stress data has random labels, measure prediction consistency
            pred_probs = clf.predict_proba(X)
            confidence_mean = float(np.mean(np.max(pred_probs, axis=1)))
            confidence_std = float(np.std(np.max(pred_probs, axis=1)))
            top5 = np.argsort(pred_probs, axis=1)[:, -5:]
            entropy = float(np.mean(-np.sum(pred_probs * np.log(pred_probs + 1e-10), axis=1)))

            clf_results = {
                "status": "✅ Inference OK",
                "samples": len(X),
                "inference_time_sec": round(inference_time, 4),
                "throughput_samples_per_sec": round(len(X) / inference_time, 1),
                "mean_prediction_confidence": round(confidence_mean, 4),
                "std_prediction_confidence": round(confidence_std, 4),
                "mean_prediction_entropy": round(entropy, 4),
                "unique_predictions": int(len(np.unique(pred_clf))),
                "total_classes_known": int(len(clf.classes_)) if hasattr(clf, 'classes_') else 0,
            }
        except Exception as e:
            clf_results = {"status": f"❌ FAILED: {str(e)}"}

        # --- Regressor evaluation ---
        try:
            start = time.time()
            y_log_pred = reg.predict(X)
            reg_inference_time = time.time() - start

            pred_reg = np.expm1(y_log_pred)
            pred_reg = np.clip(pred_reg, 0, None)  # duration can't be negative

            mae = mean_absolute_error(y_true_reg, pred_reg)
            mse = mean_squared_error(y_true_reg, pred_reg)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_reg, pred_reg)

            # Check for degenerate predictions
            pred_std = float(np.std(pred_reg))
            pred_range = float(np.ptp(pred_reg))
            negative_count = int(np.sum(y_log_pred < 0))

            reg_results = {
                "status": "✅ Inference OK",
                "samples": len(X),
                "inference_time_sec": round(reg_inference_time, 4),
                "MAE": round(mae, 4),
                "MSE": round(mse, 2),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "pred_mean": round(float(np.mean(pred_reg)), 4),
                "pred_std": round(pred_std, 4),
                "pred_range": round(pred_range, 4),
                "negative_pred_count": negative_count,
                "true_mean": round(float(np.mean(y_true_reg)), 4),
            }
        except Exception as e:
            reg_results = {"status": f"❌ FAILED: {str(e)}"}

        results[scenario] = {
            "classifier": clf_results,
            "regressor": reg_results,
        }

        print(f"\n  Classifier: {clf_results.get('status', 'N/A')}")
        if 'mean_prediction_confidence' in clf_results:
            print(f"    Confidence: {clf_results['mean_prediction_confidence']:.4f} ± {clf_results['std_prediction_confidence']:.4f}")
            print(f"    Entropy: {clf_results['mean_prediction_entropy']:.4f}")
            print(f"    Unique predictions: {clf_results['unique_predictions']}/{clf_results['total_classes_known']}")

        print(f"\n  Regressor: {reg_results.get('status', 'N/A')}")
        if 'MAE' in reg_results:
            print(f"    MAE={reg_results['MAE']:.4f}, RMSE={reg_results['RMSE']:.4f}, R²={reg_results['R2']:.4f}")
            print(f"    Pred range: [{reg_results['pred_mean'] - reg_results['pred_std']:.2f}, {reg_results['pred_mean'] + reg_results['pred_std']:.2f}]")

    return results


# ============================================================
# PART 2: RL Environment Stress Tests
# ============================================================

def stress_test_rl_environment():
    """Stress test the RL scheduling environment under extreme conditions."""
    from rl_pipeline.env_dataset import RLEnvDataset
    from rl_pipeline.environment import TaskSchedulingEnv, RewardWeights

    rng = np.random.default_rng(42)
    results = {}

    # --- Scenario 1: Many tasks, few machines, tight capacity ---
    print(f"\n{'='*60}")
    print(f"  RL STRESS: RESOURCE SCARCITY (32 machines, 500 tasks)")
    print(f"{'='*60}")

    n_machines = 32
    n_tasks = 500
    machines = pd.DataFrame([
        {
            "machine_id": f"m{i}",
            "cpu_capacity": float(rng.uniform(0.3, 0.6)),
            "mem_capacity": float(rng.uniform(0.3, 0.6)),
            "disk_capacity": float(rng.uniform(0.2, 0.4)),
        }
        for i in range(n_machines)
    ])

    tasks = pd.DataFrame([
        {
            "episode_id": 0,
            "task_index": i,
            "task_id": f"t{i}",
            "arrival_time": float(rng.uniform(0, 100)),
            "duration": float(rng.exponential(10)),
            "cpu_demand": float(rng.uniform(0.05, 0.3)),
            "mem_demand": float(rng.uniform(0.05, 0.25)),
            "disk_demand": float(rng.uniform(0.01, 0.1)),
            "historical_machine_id": f"m{rng.integers(0, n_machines)}",
        }
        for i in range(n_tasks)
    ])

    episodes = pd.DataFrame([
        {"episode_id": 0, "task_count": n_tasks, "start_time": 0.0,
         "end_time": 200.0, "source_kind": "stress_test"}
    ])

    dataset = RLEnvDataset(
        tasks=tasks, machines=machines, episodes=episodes,
        metadata={"source_kind": "stress_test"}
    )

    env = TaskSchedulingEnv(
        dataset=dataset,
        top_k_candidates=16,
        max_steps=2000,
        max_consecutive_defers=50,
        invalid_action_limit=50,
        machine_capacity_scale=0.35,
        randomize_on_reset=False,
    )

    # Run greedy heuristic: always pick the first feasible action
    obs, info = env.reset(options={"episode_id": 0})
    total_reward = 0.0
    step_count = 0
    feasible_assignments = 0
    defers = 0
    deadline_misses = 0
    rewards_by_type = {}

    while True:
        mask = info["action_mask"]
        # Pick first feasible candidate (not defer)
        feasible_indices = [i for i in range(len(mask) - 1) if mask[i] == 1]

        if feasible_indices:
            action = feasible_indices[0]
        else:
            action = env.defer_action
            defers += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        for key, val in info.get("reward_components", {}).items():
            rewards_by_type[key] = rewards_by_type.get(key, 0) + val

        if action != env.defer_action and info.get("was_feasible", False):
            feasible_assignments += 1
        if "missed_deadline_penalty" in info.get("reward_components", {}):
            deadline_misses += 1

        if terminated or truncated:
            break

    completion_rate = feasible_assignments / n_tasks
    results["resource_scarcity"] = {
        "machines": n_machines,
        "tasks": n_tasks,
        "capacity_scale": 0.35,
        "steps_taken": step_count,
        "total_reward": round(total_reward, 4),
        "scheduled_tasks": feasible_assignments,
        "completion_rate": round(completion_rate, 4),
        "defers": defers,
        "deadline_misses": deadline_misses,
        "terminated": terminated,
        "truncated": truncated,
        "reward_breakdown": {k: round(v, 2) for k, v in rewards_by_type.items()},
    }

    print(f"  Scheduled: {feasible_assignments}/{n_tasks} ({completion_rate:.1%})")
    print(f"  Steps: {step_count}, Defers: {defers}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Deadline misses: {deadline_misses}")

    # --- Scenario 2: Burst arrivals (all tasks arrive at t=0) ---
    print(f"\n{'='*60}")
    print(f"  RL STRESS: BURST ARRIVALS (200 tasks at t=0)")
    print(f"{'='*60}")

    n_burst = 200
    burst_tasks = pd.DataFrame([
        {
            "episode_id": 0,
            "task_index": i,
            "task_id": f"b{i}",
            "arrival_time": 0.0,
            "duration": float(rng.exponential(5)),
            "cpu_demand": float(rng.uniform(0.05, 0.2)),
            "mem_demand": float(rng.uniform(0.05, 0.15)),
            "disk_demand": float(rng.uniform(0.01, 0.05)),
            "historical_machine_id": f"m{rng.integers(0, n_machines)}",
        }
        for i in range(n_burst)
    ])

    burst_episodes = pd.DataFrame([
        {"episode_id": 0, "task_count": n_burst, "start_time": 0.0,
         "end_time": 50.0, "source_kind": "stress_burst"}
    ])

    burst_dataset = RLEnvDataset(
        tasks=burst_tasks, machines=machines, episodes=burst_episodes,
        metadata={"source_kind": "stress_burst"}
    )

    burst_env = TaskSchedulingEnv(
        dataset=burst_dataset,
        top_k_candidates=16,
        max_steps=1000,
        max_consecutive_defers=30,
        invalid_action_limit=30,
        randomize_on_reset=False,
    )

    obs, info = burst_env.reset(options={"episode_id": 0})
    burst_reward = 0.0
    burst_steps = 0
    burst_scheduled = 0
    burst_defers = 0

    while True:
        mask = info["action_mask"]
        feasible = [i for i in range(len(mask) - 1) if mask[i] == 1]
        action = feasible[0] if feasible else burst_env.defer_action
        if action == burst_env.defer_action:
            burst_defers += 1

        obs, reward, terminated, truncated, info = burst_env.step(action)
        burst_reward += reward
        burst_steps += 1

        if action != burst_env.defer_action and info.get("was_feasible", False):
            burst_scheduled += 1

        if terminated or truncated:
            break

    results["burst_arrivals"] = {
        "tasks": n_burst,
        "scheduled": burst_scheduled,
        "completion_rate": round(burst_scheduled / n_burst, 4),
        "steps": burst_steps,
        "defers": burst_defers,
        "total_reward": round(burst_reward, 4),
        "terminated": terminated,
        "truncated": truncated,
    }

    print(f"  Scheduled: {burst_scheduled}/{n_burst} ({burst_scheduled/n_burst:.1%})")
    print(f"  Steps: {burst_steps}, Defers: {burst_defers}")
    print(f"  Total reward: {burst_reward:.2f}")

    # --- Scenario 3: Single heavy machine (bottleneck) ---
    print(f"\n{'='*60}")
    print(f"  RL STRESS: SINGLE BOTTLENECK MACHINE")
    print(f"{'='*60}")

    bottleneck_machines = pd.DataFrame([
        {"machine_id": "m_big", "cpu_capacity": 2.0, "mem_capacity": 2.0, "disk_capacity": 2.0},
        {"machine_id": "m_tiny", "cpu_capacity": 0.05, "mem_capacity": 0.05, "disk_capacity": 0.05},
    ])

    n_bottleneck = 100
    bottleneck_tasks = pd.DataFrame([
        {
            "episode_id": 0,
            "task_index": i,
            "task_id": f"bn{i}",
            "arrival_time": float(i * 0.5),
            "duration": float(rng.exponential(3)),
            "cpu_demand": float(rng.uniform(0.1, 0.5)),
            "mem_demand": float(rng.uniform(0.1, 0.4)),
            "disk_demand": float(rng.uniform(0.01, 0.1)),
            "historical_machine_id": "m_big",
        }
        for i in range(n_bottleneck)
    ])

    bottleneck_episodes = pd.DataFrame([
        {"episode_id": 0, "task_count": n_bottleneck, "start_time": 0.0,
         "end_time": 60.0, "source_kind": "stress_bottleneck"}
    ])

    bn_dataset = RLEnvDataset(
        tasks=bottleneck_tasks, machines=bottleneck_machines, episodes=bottleneck_episodes,
        metadata={"source_kind": "stress_bottleneck"}
    )

    bn_env = TaskSchedulingEnv(
        dataset=bn_dataset,
        top_k_candidates=4,
        max_steps=500,
        max_consecutive_defers=20,
        invalid_action_limit=20,
        randomize_on_reset=False,
    )

    obs, info = bn_env.reset(options={"episode_id": 0})
    bn_reward = 0.0
    bn_steps = 0
    bn_scheduled = 0
    bn_defers = 0

    while True:
        mask = info["action_mask"]
        feasible = [i for i in range(len(mask) - 1) if mask[i] == 1]
        action = feasible[0] if feasible else bn_env.defer_action
        if action == bn_env.defer_action:
            bn_defers += 1

        obs, reward, terminated, truncated, info = bn_env.step(action)
        bn_reward += reward
        bn_steps += 1

        if action != bn_env.defer_action and info.get("was_feasible", False):
            bn_scheduled += 1

        if terminated or truncated:
            break

    results["bottleneck"] = {
        "machines": 2,
        "tasks": n_bottleneck,
        "scheduled": bn_scheduled,
        "completion_rate": round(bn_scheduled / n_bottleneck, 4),
        "steps": bn_steps,
        "defers": bn_defers,
        "total_reward": round(bn_reward, 4),
        "terminated": terminated,
        "truncated": truncated,
    }

    print(f"  Scheduled: {bn_scheduled}/{n_bottleneck} ({bn_scheduled/n_bottleneck:.1%})")
    print(f"  Steps: {bn_steps}, Defers: {bn_defers}")
    print(f"  Total reward: {bn_reward:.2f}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  AI DEADLINE-AWARE OS SCHEDULER — COMPREHENSIVE STRESS TEST")
    print("=" * 70)

    all_results = {}

    # Part 1: XGBoost
    print("\n\n" + "▓" * 70)
    print("  PART 1: XGBoost Model Stress Tests")
    print("▓" * 70)

    clf, reg = load_xgboost_models()
    if clf is not None and reg is not None:
        all_results["xgboost"] = stress_test_xgboost(clf, reg)
    else:
        all_results["xgboost"] = {"status": "SKIPPED — models not found"}

    # Part 2: RL Environment
    print("\n\n" + "▓" * 70)
    print("  PART 2: RL Environment Stress Tests")
    print("▓" * 70)

    try:
        all_results["rl_environment"] = stress_test_rl_environment()
    except Exception as e:
        all_results["rl_environment"] = {"status": f"FAILED: {str(e)}"}
        import traceback
        traceback.print_exc()

    # Save results
    output_file = RESULT_PATH / "stress_test_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'='*70}")
    print(f"  STRESS TEST COMPLETE — Results saved to: {output_file}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    main()
