from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .env_dataset import RLEnvDataset
from .environment import TaskSchedulingEnv

try:
    from data_preprocessing.pipeline_config import RL_ENV_DATASET_JSON_GZ, RL_ENV_VALIDATION_REPORT_JSON
except ImportError:
    RL_ENV_DATASET_JSON_GZ = Path("data/interim/rl_env_dataset.json.gz")
    RL_ENV_VALIDATION_REPORT_JSON = Path("data/reports/rl_env_validation.json")


def _build_validation_dataset() -> RLEnvDataset:
    machines = pd.DataFrame(
        [
            {"machine_id": "m_big", "cpu_capacity": 1.0, "mem_capacity": 1.0, "disk_capacity": 1.0},
            {"machine_id": "m_small", "cpu_capacity": 0.25, "mem_capacity": 0.25, "disk_capacity": 0.25},
        ]
    )
    tasks = pd.DataFrame(
        [
            {
                "episode_id": 0,
                "task_index": 0,
                "task_id": "t0",
                "arrival_time": 0.0,
                "duration": 4.0,
                "cpu_demand": 0.60,
                "mem_demand": 0.50,
                "disk_demand": 0.20,
                "historical_machine_id": "m_big",
            },
            {
                "episode_id": 0,
                "task_index": 1,
                "task_id": "t1",
                "arrival_time": 1.0,
                "duration": 2.0,
                "cpu_demand": 0.20,
                "mem_demand": 0.10,
                "disk_demand": 0.10,
                "historical_machine_id": "m_small",
            },
            {
                "episode_id": 0,
                "task_index": 2,
                "task_id": "t2",
                "arrival_time": 4.0,
                "duration": 1.0,
                "cpu_demand": 0.15,
                "mem_demand": 0.10,
                "disk_demand": 0.05,
                "historical_machine_id": "m_big",
            },
        ]
    )
    episodes = pd.DataFrame(
        [{"episode_id": 0, "task_count": 3, "start_time": 0.0, "end_time": 5.0, "source_kind": "validation"}]
    )
    return RLEnvDataset(tasks=tasks, machines=machines, episodes=episodes, metadata={"source_kind": "validation"})


def _assert(condition: bool, message: str) -> tuple[bool, str | None]:
    return bool(condition), None if condition else message


def _candidate_consistency(info: dict[str, Any]) -> tuple[bool, str | None]:
    action_mask = np.asarray(info["action_mask"])
    candidate_machine_ids = list(info["candidate_machine_ids"])
    if int(action_mask.sum()) <= 0:
        return False, "Action mask has no enabled actions."

    for index, machine_id in enumerate(candidate_machine_ids):
        masked_on = bool(action_mask[index])
        if masked_on and machine_id is None:
            return False, f"Action slot {index} is enabled but candidate_machine_ids[{index}] is None."
        if (not masked_on) and machine_id is not None:
            return False, f"Action slot {index} is disabled but candidate_machine_ids[{index}] is populated."
    return True, None


def validate_environment_behavior() -> dict[str, Any]:
    dataset = _build_validation_dataset()
    env = TaskSchedulingEnv(
        dataset=dataset,
        top_k_candidates=4,
        max_steps=20,
        max_consecutive_defers=6,
        invalid_action_limit=6,
        randomize_on_reset=False,
    )

    report: dict[str, Any] = {"checks": {}, "reward_trace": {}, "manual_rollout": []}

    observation, reset_info = env.reset(options={"episode_id": 0})
    report["checks"]["mask_valid_at_reset"] = {}
    ok, error = _candidate_consistency(reset_info)
    report["checks"]["mask_valid_at_reset"]["passed"] = ok
    report["checks"]["mask_valid_at_reset"]["error"] = error

    infeasible_action = next(
        index for index, features in enumerate(observation["candidate_features"]) if features[0] > 0 and features[1] < 0.5
    )
    feasible_action = next(
        index for index, features in enumerate(observation["candidate_features"]) if features[0] > 0 and features[1] > 0.5
    )

    before_overload_time = env.current_time
    _, overload_reward, _, _, overload_info = env.step(infeasible_action)
    ok, error = _assert(
        float(overload_info["current_time"]) == float(before_overload_time),
        "Overload action changed time even though no arrival/completion event happened.",
    )
    report["checks"]["time_evolution_after_overload"] = {"passed": ok, "error": error}
    report["reward_trace"]["overload_action_reward"] = overload_reward
    report["manual_rollout"].append(
        {
            "step": "overload_assignment",
            "reward": overload_reward,
            "decision_time": overload_info["decision_time"],
            "current_time": overload_info["current_time"],
            "task_id": overload_info["task_id"],
            "reward_components": overload_info["reward_components"],
        }
    )

    feasible_action = next(
        index for index, features in enumerate(env._candidate_features) if features[0] > 0 and features[1] > 0.5
    )
    feasible_machine_index = env._candidate_machine_indices[feasible_action]
    before_feasible_residual = env.machine_residual[feasible_machine_index].copy()
    _, feasible_reward, _, _, feasible_info = env.step(feasible_action)
    after_feasible_residual = env.machine_residual[feasible_machine_index].copy()

    ok, error = _assert(np.any(after_feasible_residual < before_feasible_residual), "Feasible assignment did not consume capacity.")
    report["checks"]["resource_accounting_after_allocation"] = {"passed": ok, "error": error}

    valid_event_times = {1.0, 4.0}
    ok, error = _assert(
        float(feasible_info["current_time"]) in valid_event_times,
        f"Time jumped to {feasible_info['current_time']} instead of the next arrival or completion.",
    )
    report["checks"]["time_evolution_after_feasible_action"] = {"passed": ok, "error": error}

    report["reward_trace"]["feasible_action_reward"] = feasible_reward
    report["manual_rollout"].append(
        {
            "step": "feasible_assignment",
            "reward": feasible_reward,
            "decision_time": feasible_info["decision_time"],
            "current_time": feasible_info["current_time"],
            "task_id": feasible_info["task_id"],
            "reward_components": feasible_info["reward_components"],
        }
    )

    before_defer_queue = list(env.pending_queue)
    before_defer_time = env.current_time
    _, defer_reward, _, _, defer_info = env.step(env.defer_action)
    after_defer_queue = list(env.pending_queue)
    next_event_time = env._next_external_event_time()
    ok, error = _assert(
        before_defer_queue != after_defer_queue
        or float(defer_info["current_time"]) > float(before_defer_time)
        or next_event_time is None,
        "Defer neither changed queue order nor advanced to a real external event.",
    )
    report["checks"]["queue_dynamics_after_defer"] = {"passed": ok, "error": error}
    ok, error = _candidate_consistency(defer_info)
    report["checks"]["mask_valid_after_defer"] = {"passed": ok, "error": error}
    report["reward_trace"]["defer_action_reward"] = defer_reward
    report["manual_rollout"].append(
        {
            "step": "defer_action",
            "reward": defer_reward,
            "decision_time": defer_info["decision_time"],
            "current_time": defer_info["current_time"],
            "task_id": defer_info["task_id"],
            "reward_components": defer_info["reward_components"],
        }
    )

    feasible_again = next(
        index
        for index, features in enumerate(env._candidate_features)
        if features[0] > 0 and features[1] > 0.5
    )
    env.step(feasible_again)
    while env.running_jobs:
        next_event_time = env._next_external_event_time()
        if next_event_time is None:
            break
        env._advance_clock(next_event_time)

    restored_residual = env.machine_residual[feasible_machine_index].copy()
    ok, error = _assert(
        np.allclose(restored_residual, env.machine_capacities[feasible_machine_index]),
        "Machine capacity did not fully restore after job completion.",
    )
    report["checks"]["resource_accounting_after_completion"] = {"passed": ok, "error": error}

    ok, error = _assert(
        feasible_reward > 0 and overload_reward < defer_reward < 0,
        "Reward ordering is wrong; expected feasible > 0, overload < defer < 0.",
    )
    report["checks"]["reward_signal_behavior"] = {"passed": ok, "error": error}

    report["summary"] = {
        "passed_all_checks": all(check["passed"] for check in report["checks"].values()),
        "feasible_reward": feasible_reward,
        "overload_reward": overload_reward,
        "defer_reward": defer_reward,
        "observed_episode_length": len(dataset.tasks),
    }
    return report


def validate_env_dataset_scale(dataset_path: str | Path) -> dict[str, Any]:
    from .env_dataset import load_env_dataset

    dataset = load_env_dataset(dataset_path)
    episode_count = int(len(dataset.episodes))
    min_task_count = int(dataset.episodes["task_count"].min()) if not dataset.episodes.empty else 0
    return {
        "dataset_path": str(dataset_path),
        "episode_count": episode_count,
        "minimum_tasks_per_episode": min_task_count,
        "meets_scale_target": bool(episode_count >= 100 and min_task_count >= 128),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research-grade behavior validation for the RL scheduler environment.")
    parser.add_argument("--dataset", type=Path, default=RL_ENV_DATASET_JSON_GZ)
    parser.add_argument("--output", type=Path, default=RL_ENV_VALIDATION_REPORT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    behavior_report = validate_environment_behavior()
    report = {
        "behavior_validation": behavior_report,
        "dataset_scale_validation": validate_env_dataset_scale(args.dataset) if args.dataset.exists() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved RL environment validation report: {args.output}")
    print(f"Behavior checks passed: {behavior_report['summary']['passed_all_checks']}")
    if report["dataset_scale_validation"] is not None:
        print(f"Dataset scale target met: {report['dataset_scale_validation']['meets_scale_target']}")


if __name__ == "__main__":
    main()
