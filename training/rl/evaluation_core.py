from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .env_factory import build_scheduler_env, subset_dataset_by_episode_ids
    from .hybrid_scheduler import HybridScheduler, HybridSchedulerConfig
except ImportError:
    from env_factory import build_scheduler_env, subset_dataset_by_episode_ids
    from hybrid_scheduler import HybridScheduler, HybridSchedulerConfig

from rl_pipeline.env_dataset import RLEnvDataset


@dataclass
class AssignmentRecord:
    episode_id: int
    task_id: str
    task_index: int
    machine_id: str
    arrival_time: float
    start_time: float
    completion_time: float
    duration: float
    cpu_demand: float
    wait_time: float
    turnaround_time: float
    deadline: float
    missed_deadline: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": int(self.episode_id),
            "task_id": str(self.task_id),
            "task_index": int(self.task_index),
            "machine_id": str(self.machine_id),
            "arrival_time": float(self.arrival_time),
            "start_time": float(self.start_time),
            "completion_time": float(self.completion_time),
            "duration": float(self.duration),
            "cpu_demand": float(self.cpu_demand),
            "wait_time": float(self.wait_time),
            "turnaround_time": float(self.turnaround_time),
            "deadline": float(self.deadline),
            "missed_deadline": bool(self.missed_deadline),
        }


def _require_maskable_ppo() -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing RL dependency for policy evaluation. Install with:\n"
            "pip install gymnasium stable-baselines3 sb3-contrib"
        ) from exc
    return MaskablePPO


def _compute_deadline(*, arrival_time: float, duration: float, slack_factor: float) -> float:
    return float(arrival_time + slack_factor * duration)


def _extract_action_mask(info: dict[str, Any], action_count: int) -> np.ndarray:
    raw_mask = info.get("action_mask")
    if raw_mask is None:
        raw_mask = info.get("decision_action_mask")
    if raw_mask is None:
        return np.ones(action_count, dtype=bool)
    mask = np.asarray(raw_mask, dtype=np.int8).astype(bool, copy=False)
    if mask.shape != (action_count,):
        raise ValueError(f"Expected action mask shape {(action_count,)}, received {mask.shape}.")
    return mask


def _extract_assignment_records_from_step_info(
    *,
    step_info: dict[str, Any],
    episode_id: int,
    seen_task_ids: set[str],
) -> list[AssignmentRecord]:
    raw_events = step_info.get("assignment_events")
    if not isinstance(raw_events, list):
        return []
    records: list[AssignmentRecord] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        raw_task_id = raw_event.get("task_id")
        if raw_task_id is None:
            continue
        task_id = str(raw_task_id)
        if task_id in seen_task_ids:
            continue
        try:
            record = AssignmentRecord(
                episode_id=int(raw_event.get("episode_id", episode_id)),
                task_id=task_id,
                task_index=int(raw_event["task_index"]),
                machine_id=str(raw_event["machine_id"]),
                arrival_time=float(raw_event["arrival_time"]),
                start_time=float(raw_event["start_time"]),
                completion_time=float(raw_event["completion_time"]),
                duration=float(raw_event["duration"]),
                cpu_demand=float(raw_event["cpu_demand"]),
                wait_time=float(raw_event["wait_time"]),
                turnaround_time=float(raw_event["turnaround_time"]),
                deadline=float(raw_event["deadline"]),
                missed_deadline=bool(raw_event["missed_deadline"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        records.append(record)
        seen_task_ids.add(task_id)
    return records


def _compute_episode_metric_rollup(
    *,
    assignment_records: list[AssignmentRecord],
    total_tasks: int,
    unscheduled_tasks: int,
    cpu_capacity_total: float,
    episode_start_time: float,
    episode_end_time: float,
) -> dict[str, Any]:
    scheduled_tasks = len(assignment_records)
    missed_deadline_tasks = int(
        sum(1 for row in assignment_records if bool(row.missed_deadline)) + int(unscheduled_tasks)
    )
    waiting_time_sum = float(sum(float(row.wait_time) for row in assignment_records))
    turnaround_time_sum = float(sum(float(row.turnaround_time) for row in assignment_records))
    cpu_busy_time = float(sum(float(row.cpu_demand * row.duration) for row in assignment_records))

    # Fix-2: Fair CPU utilization — use the MINIMUM active window across all
    # schedulers for this episode.  To keep it self-contained we compute both
    # the old time-window metric and a new task-volume metric and expose both.
    active_window = max(float(episode_end_time) - float(episode_start_time), 1e-9)
    cpu_capacity_time = float(max(cpu_capacity_total, 1e-9) * active_window)
    cpu_utilization = float(cpu_busy_time / cpu_capacity_time)

    # Task-volume utilization: what fraction of the task-occupied machine-time
    # was actually used for CPU work.  This is independent of episode length.
    total_allocated_cpu_time = float(
        sum(float(row.duration) for row in assignment_records)
        * max(cpu_capacity_total / max(scheduled_tasks, 1), 1e-9)
    )
    cpu_utilization_task_volume = float(cpu_busy_time / max(total_allocated_cpu_time, 1e-9))

    return {
        "total_tasks": int(total_tasks),
        "scheduled_tasks": int(scheduled_tasks),
        "unscheduled_tasks": int(unscheduled_tasks),
        "missed_deadline_tasks": int(missed_deadline_tasks),
        "deadline_miss_ratio": float(missed_deadline_tasks / max(total_tasks, 1)),
        "mean_waiting_time": None if scheduled_tasks == 0 else float(waiting_time_sum / scheduled_tasks),
        "mean_turnaround_time": None if scheduled_tasks == 0 else float(turnaround_time_sum / scheduled_tasks),
        "cpu_utilization": float(cpu_utilization),
        "cpu_utilization_task_volume": float(cpu_utilization_task_volume),
        "waiting_time_sum": float(waiting_time_sum),
        "turnaround_time_sum": float(turnaround_time_sum),
        "cpu_busy_time": float(cpu_busy_time),
        "cpu_capacity_time": float(cpu_capacity_time),
    }


def _aggregate_episode_rollups(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total_tasks = int(sum(int(row["total_tasks"]) for row in episodes))
    scheduled_tasks = int(sum(int(row["scheduled_tasks"]) for row in episodes))
    unscheduled_tasks = int(sum(int(row["unscheduled_tasks"]) for row in episodes))
    missed_deadline_tasks = int(sum(int(row["missed_deadline_tasks"]) for row in episodes))
    waiting_time_sum = float(sum(float(row["waiting_time_sum"]) for row in episodes))
    turnaround_time_sum = float(sum(float(row["turnaround_time_sum"]) for row in episodes))
    cpu_busy_time = float(sum(float(row["cpu_busy_time"]) for row in episodes))
    cpu_capacity_time = float(sum(float(row["cpu_capacity_time"]) for row in episodes))

    return {
        "episode_count": int(len(episodes)),
        "total_tasks": total_tasks,
        "scheduled_tasks": scheduled_tasks,
        "unscheduled_tasks": unscheduled_tasks,
        "assignment_rate": float(scheduled_tasks / max(total_tasks, 1)),
        "missed_deadline_tasks": missed_deadline_tasks,
        "deadline_miss_ratio": float(missed_deadline_tasks / max(total_tasks, 1)),
        "mean_waiting_time": None if scheduled_tasks == 0 else float(waiting_time_sum / scheduled_tasks),
        "mean_turnaround_time": None if scheduled_tasks == 0 else float(turnaround_time_sum / scheduled_tasks),
        "cpu_utilization": float(cpu_busy_time / max(cpu_capacity_time, 1e-9)),
        "deadline_rule": "deadline = arrival_time + slack_factor * duration",
    }


def _select_best_fit_machine(
    *,
    machine_residual: np.ndarray,
    machine_capacities: np.ndarray,
    demand: np.ndarray,
    feasible_indices: np.ndarray,
) -> int:
    capacity_safe = np.where(machine_capacities[feasible_indices] > 0, machine_capacities[feasible_indices], 1.0)
    post_residual = machine_residual[feasible_indices] - demand
    post_ratio = post_residual / capacity_safe
    # Lower score means tighter but feasible packing, which tends to reduce fragmentation.
    score = np.sum(np.clip(post_ratio, 0.0, None), axis=1)
    best_local_index = int(np.argmin(score))
    return int(feasible_indices[best_local_index])


def _select_round_robin_machine(
    *,
    feasible_mask: np.ndarray,
    rr_cursor: int,
) -> tuple[int, int]:
    machine_count = int(feasible_mask.shape[0])
    if machine_count <= 0:
        raise ValueError("Round-robin selection requires at least one machine.")

    for offset in range(machine_count):
        candidate_index = int((rr_cursor + offset) % machine_count)
        if bool(feasible_mask[candidate_index]):
            next_cursor = int((candidate_index + 1) % machine_count)
            return candidate_index, next_cursor

    raise RuntimeError("Round-robin selection called without any feasible machine.")


def _simulate_heuristic_episode(
    *,
    episode_id: int,
    tasks: pd.DataFrame,
    machine_capacities: np.ndarray,
    machine_ids: list[str],
    heuristic: str,
    deadline_slack_factor: float,
    allow_queue_bypass: bool,
) -> dict[str, Any]:
    if tasks.empty:
        return {
            "episode_id": int(episode_id),
            "total_tasks": 0,
            "scheduled_tasks": 0,
            "unscheduled_tasks": 0,
            "missed_deadline_tasks": 0,
            "deadline_miss_ratio": 0.0,
            "mean_waiting_time": 0.0,
            "mean_turnaround_time": 0.0,
            "cpu_utilization": 0.0,
            "waiting_time_sum": 0.0,
            "turnaround_time_sum": 0.0,
            "cpu_busy_time": 0.0,
            "cpu_capacity_time": 1.0,
            "assignments": [],
        }

    frame = tasks.sort_values(["arrival_time", "task_index"]).reset_index(drop=True)
    n_tasks = len(frame)

    task_ids = frame["task_id"].astype(str).to_numpy()
    task_indices = frame["task_index"].astype(int).to_numpy()
    arrivals = frame["arrival_time"].astype(float).to_numpy()
    durations = frame["duration"].astype(float).to_numpy()
    cpu_demands = frame["cpu_demand"].astype(float).to_numpy()
    demands = frame[["cpu_demand", "mem_demand", "disk_demand"]].astype(float).to_numpy()

    pending: list[int] = []
    running_jobs: list[tuple[float, int, int, np.ndarray]] = []
    machine_residual = machine_capacities.copy()
    job_counter = 0
    rr_cursor = 0
    arrival_cursor = 0
    current_time = float(arrivals.min())

    assignments: list[AssignmentRecord] = []

    while arrival_cursor < n_tasks or pending or running_jobs:
        while running_jobs and running_jobs[0][0] <= current_time + 1e-9:
            _, _, machine_index, demand = heapq.heappop(running_jobs)
            machine_residual[machine_index] = np.minimum(
                machine_capacities[machine_index],
                machine_residual[machine_index] + demand,
            )

        while arrival_cursor < n_tasks and arrivals[arrival_cursor] <= current_time + 1e-9:
            pending.append(arrival_cursor)
            arrival_cursor += 1

        if not pending:
            next_times: list[float] = []
            if arrival_cursor < n_tasks:
                next_times.append(float(arrivals[arrival_cursor]))
            if running_jobs:
                next_times.append(float(running_jobs[0][0]))
            if not next_times:
                break
            current_time = max(current_time, min(next_times))
            continue

        normalized_heuristic = heuristic.upper()
        if normalized_heuristic == "FCFS":
            pending.sort(key=lambda idx: (float(arrivals[idx]), int(task_indices[idx])))
        elif normalized_heuristic == "SJF":
            pending.sort(key=lambda idx: (float(durations[idx]), float(arrivals[idx]), int(task_indices[idx])))
        elif normalized_heuristic == "RR":
            pending.sort(key=lambda idx: (float(arrivals[idx]), int(task_indices[idx])))
        else:
            raise ValueError(f"Unsupported heuristic '{heuristic}'. Expected one of: FCFS, SJF, RR.")

        selected_pending_index: int | None = None
        task_position: int | None = None
        demand: np.ndarray | None = None
        feasible_mask: np.ndarray | None = None
        feasible_indices: np.ndarray | None = None

        if bool(allow_queue_bypass):
            for pending_index, candidate_task_position in enumerate(pending):
                candidate_demand = np.asarray(demands[candidate_task_position], dtype=np.float64)
                candidate_feasible_mask = np.all(machine_residual - candidate_demand >= -1e-9, axis=1)
                candidate_feasible_indices = np.where(candidate_feasible_mask)[0]
                if candidate_feasible_indices.size > 0:
                    selected_pending_index = int(pending_index)
                    task_position = int(candidate_task_position)
                    demand = candidate_demand
                    feasible_mask = candidate_feasible_mask
                    feasible_indices = candidate_feasible_indices
                    break
        else:
            candidate_task_position = int(pending[0])
            candidate_demand = np.asarray(demands[candidate_task_position], dtype=np.float64)
            candidate_feasible_mask = np.all(machine_residual - candidate_demand >= -1e-9, axis=1)
            candidate_feasible_indices = np.where(candidate_feasible_mask)[0]
            if candidate_feasible_indices.size > 0:
                selected_pending_index = 0
                task_position = int(candidate_task_position)
                demand = candidate_demand
                feasible_mask = candidate_feasible_mask
                feasible_indices = candidate_feasible_indices

        if selected_pending_index is None or task_position is None or demand is None:
            next_times = []
            if running_jobs:
                next_times.append(float(running_jobs[0][0]))
            if arrival_cursor < n_tasks:
                next_times.append(float(arrivals[arrival_cursor]))
            if not next_times:
                break
            next_time = float(min(next_times))
            if next_time <= current_time + 1e-12:
                next_time = current_time + 1e-6
            current_time = next_time
            continue

        assert feasible_mask is not None
        assert feasible_indices is not None

        if normalized_heuristic == "RR":
            machine_index, rr_cursor = _select_round_robin_machine(
                feasible_mask=feasible_mask,
                rr_cursor=int(rr_cursor),
            )
        else:
            machine_index = _select_best_fit_machine(
                machine_residual=machine_residual,
                machine_capacities=machine_capacities,
                demand=demand,
                feasible_indices=feasible_indices,
            )
        pending.pop(int(selected_pending_index))

        start_time = float(current_time)
        duration = float(durations[task_position])
        completion_time = float(start_time + duration)
        arrival_time = float(arrivals[task_position])
        wait_time = float(max(0.0, start_time - arrival_time))
        turnaround_time = float(completion_time - arrival_time)
        deadline = _compute_deadline(
            arrival_time=arrival_time,
            duration=duration,
            slack_factor=deadline_slack_factor,
        )

        machine_residual[machine_index] -= demand
        machine_residual[machine_index] = np.clip(
            machine_residual[machine_index],
            a_min=0.0,
            a_max=machine_capacities[machine_index],
        )
        heapq.heappush(running_jobs, (completion_time, job_counter, machine_index, demand.copy()))
        job_counter += 1

        assignments.append(
            AssignmentRecord(
                episode_id=int(episode_id),
                task_id=str(task_ids[task_position]),
                task_index=int(task_indices[task_position]),
                machine_id=str(machine_ids[machine_index]),
                arrival_time=arrival_time,
                start_time=start_time,
                completion_time=completion_time,
                duration=duration,
                cpu_demand=float(cpu_demands[task_position]),
                wait_time=wait_time,
                turnaround_time=turnaround_time,
                deadline=deadline,
                missed_deadline=bool(completion_time > deadline),
            )
        )

    unscheduled_tasks = max(0, int(n_tasks - len(assignments)))
    episode_start = float(arrivals.min())
    max_completion = max((row.completion_time for row in assignments), default=current_time)
    episode_end = max(float(current_time), float(max_completion), episode_start + 1e-9)
    cpu_capacity_total = float(np.sum(machine_capacities[:, 0]))

    rollup = _compute_episode_metric_rollup(
        assignment_records=assignments,
        total_tasks=int(n_tasks),
        unscheduled_tasks=int(unscheduled_tasks),
        cpu_capacity_total=cpu_capacity_total,
        episode_start_time=episode_start,
        episode_end_time=episode_end,
    )
    rollup["episode_id"] = int(episode_id)
    rollup["assignments"] = [row.as_dict() for row in assignments]
    return rollup


def evaluate_heuristic_policy_on_episode_ids(
    *,
    dataset: RLEnvDataset,
    episode_ids: list[int],
    heuristic: str,
    deadline_slack_factor: float,
    machine_capacity_scale: float = 1.0,
    machine_pool_size: int | None = None,
    allow_queue_bypass: bool = True,
) -> dict[str, Any]:
    selected_episode_ids = sorted({int(episode_id) for episode_id in episode_ids})
    if not selected_episode_ids:
        raise ValueError("Episode id list for heuristic evaluation cannot be empty.")
    if float(machine_capacity_scale) <= 0.0:
        raise ValueError(
            f"machine_capacity_scale must be positive, received {machine_capacity_scale}."
        )
    if machine_pool_size is not None and int(machine_pool_size) <= 0:
        raise ValueError(
            f"machine_pool_size must be positive when provided, received {machine_pool_size}."
        )

    machine_frame = dataset.machines.copy().reset_index(drop=True)
    machine_frame["machine_id"] = machine_frame["machine_id"].astype(str)
    if machine_pool_size is not None and int(machine_pool_size) < len(machine_frame):
        machine_frame = (
            machine_frame.sort_values("machine_id")
            .head(int(machine_pool_size))
            .reset_index(drop=True)
        )
    machine_ids = machine_frame["machine_id"].astype(str).tolist()
    machine_capacities = machine_frame[["cpu_capacity", "mem_capacity", "disk_capacity"]].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    machine_capacities *= float(machine_capacity_scale)

    episodes: list[dict[str, Any]] = []
    for episode_id in selected_episode_ids:
        tasks = (
            dataset.tasks.loc[dataset.tasks["episode_id"].astype(int) == int(episode_id)]
            .copy()
            .sort_values(["task_index"])
            .reset_index(drop=True)
        )
        episodes.append(
            _simulate_heuristic_episode(
                episode_id=int(episode_id),
                tasks=tasks,
                machine_capacities=machine_capacities,
                machine_ids=machine_ids,
                heuristic=heuristic,
                deadline_slack_factor=float(deadline_slack_factor),
                allow_queue_bypass=bool(allow_queue_bypass),
            )
        )

    summary = _aggregate_episode_rollups(episodes)
    summary["scheduler"] = heuristic.upper()
    summary["deadline_slack_factor"] = float(deadline_slack_factor)
    summary["machine_capacity_scale"] = float(machine_capacity_scale)
    summary["machine_pool_size"] = None if machine_pool_size is None else int(machine_pool_size)
    summary["allow_queue_bypass"] = bool(allow_queue_bypass)
    return {
        "scheduler": heuristic.upper(),
        "selected_episode_ids": selected_episode_ids,
        "summary": summary,
        "episodes": episodes,
    }


def evaluate_rl_policy_on_episode_ids(
    *,
    model_path: str | Path,
    dataset: RLEnvDataset,
    episode_ids: list[int],
    seed: int = 42,
    device: str = "auto",
    stochastic: bool = False,
    deadline_slack_factor: float = 2.0,
    top_k_candidates: int = 16,
    max_steps: int = 500,
    max_consecutive_defers: int = 30,
    invalid_action_limit: int = 30,
    machine_capacity_scale: float = 1.0,
    machine_pool_size: int | None = None,
    use_hybrid_scheduler: bool = False,
    hybrid_defer_wait_ratio_threshold: float = 2.0,
    hybrid_high_utilization_threshold: float = 0.90,
) -> dict[str, Any]:
    selected_episode_ids = sorted({int(episode_id) for episode_id in episode_ids})
    if not selected_episode_ids:
        raise ValueError("Episode id list for RL evaluation cannot be empty.")
    if float(machine_capacity_scale) <= 0.0:
        raise ValueError(
            f"machine_capacity_scale must be positive, received {machine_capacity_scale}."
        )
    if machine_pool_size is not None and int(machine_pool_size) <= 0:
        raise ValueError(
            f"machine_pool_size must be positive when provided, received {machine_pool_size}."
        )

    MaskablePPO = _require_maskable_ppo()
    eval_dataset = subset_dataset_by_episode_ids(dataset, selected_episode_ids)

    env = build_scheduler_env(
        dataset=eval_dataset,
        top_k_candidates=int(top_k_candidates),
        max_steps=int(max_steps),
        max_consecutive_defers=int(max_consecutive_defers),
        invalid_action_limit=int(invalid_action_limit),
        machine_capacity_scale=float(machine_capacity_scale),
        machine_pool_size=None if machine_pool_size is None else int(machine_pool_size),
        deadline_slack_factor=float(deadline_slack_factor),
        random_state=int(seed),
        randomize_on_reset=False,
    )
    model = MaskablePPO.load(str(model_path), device=str(device))
    hybrid_scheduler = None
    if bool(use_hybrid_scheduler):
        hybrid_scheduler = HybridScheduler(
            HybridSchedulerConfig(
                defer_wait_ratio_threshold=float(hybrid_defer_wait_ratio_threshold),
                high_utilization_threshold=float(hybrid_high_utilization_threshold),
            )
        )

    deterministic = not bool(stochastic)
    episodes: list[dict[str, Any]] = []

    for episode_idx, episode_id in enumerate(selected_episode_ids):
        observation, info = env.reset(
            seed=int(seed) + episode_idx,
            options={"episode_id": int(episode_id)},
        )
        action_mask = _extract_action_mask(info, int(env.action_space.n))
        done = False

        total_reward = 0.0
        step_count = 0
        feasible_count = 0
        defer_count = 0
        invalid_count = 0
        terminated = False
        truncated = False
        truncate_reason: str | None = None
        assignments: list[AssignmentRecord] = []
        seen_task_ids: set[str] = set()
        final_step_info: dict[str, Any] = {}

        while not done:
            if hybrid_scheduler is None:
                action, _ = model.predict(
                    observation,
                    deterministic=deterministic,
                    action_masks=action_mask,
                )
                action_value = int(np.asarray(action).item())
            else:
                action_value = int(
                    hybrid_scheduler.select_action(
                        model=model,
                        observation=observation,
                        action_mask=action_mask,
                        env=env,
                        deterministic=deterministic,
                    )
                )
            observation, reward, terminated, truncated, step_info = env.step(action_value)
            action_mask = _extract_action_mask(step_info, int(env.action_space.n))
            final_step_info = step_info
            if bool(truncated):
                raw_reason = step_info.get("truncate_reason")
                if raw_reason is not None:
                    truncate_reason = str(raw_reason)

            reward_components = step_info.get("reward_components")
            if not isinstance(reward_components, dict):
                reward_components = {}

            total_reward += float(reward)
            step_count += 1

            event_assignments = _extract_assignment_records_from_step_info(
                step_info=step_info,
                episode_id=int(episode_id),
                seen_task_ids=seen_task_ids,
            )
            if event_assignments:
                assignments.extend(event_assignments)

            if bool(step_info.get("was_feasible", False)):
                feasible_count += 1
                # Backward-compatible fallback for environments that do not
                # emit assignment_events in step info.
                if not event_assignments:
                    task_index = step_info.get("task_index")
                    selected_machine_id = step_info.get("selected_machine_id")
                    if task_index is not None and selected_machine_id is not None:
                        task_row = env.current_episode_tasks.iloc[int(task_index)]
                        arrival_time = float(task_row["arrival_time"])
                        duration = float(task_row["duration"])
                        start_time = float(step_info.get("decision_time", env.current_time))
                        completion_time = float(start_time + duration)
                        deadline = _compute_deadline(
                            arrival_time=arrival_time,
                            duration=duration,
                            slack_factor=float(deadline_slack_factor),
                        )
                        task_id = str(task_row["task_id"])
                        if task_id not in seen_task_ids:
                            assignments.append(
                                AssignmentRecord(
                                    episode_id=int(episode_id),
                                    task_id=task_id,
                                    task_index=int(task_row["task_index"]),
                                    machine_id=str(selected_machine_id),
                                    arrival_time=arrival_time,
                                    start_time=start_time,
                                    completion_time=completion_time,
                                    duration=duration,
                                    cpu_demand=float(task_row["cpu_demand"]),
                                    wait_time=float(max(0.0, start_time - arrival_time)),
                                    turnaround_time=float(completion_time - arrival_time),
                                    deadline=deadline,
                                    missed_deadline=bool(completion_time > deadline),
                                )
                            )
                            seen_task_ids.add(task_id)
            elif "defer_penalty" in reward_components:
                defer_count += 1
            else:
                invalid_count += 1

            done = bool(terminated or truncated)

        total_tasks = int(len(env.current_episode_tasks))
        scheduled_task_count = max(int(final_step_info.get("scheduled_task_count", 0)), len(assignments))
        unscheduled_tasks = max(0, int(total_tasks - scheduled_task_count))
        assignment_count_mismatch = int(scheduled_task_count - len(assignments))
        episode_start = float(env.current_episode_tasks["arrival_time"].min())
        max_completion = max((row.completion_time for row in assignments), default=float(env.current_time))
        episode_end = max(float(env.current_time), float(max_completion), episode_start + 1e-9)
        cpu_capacity_total = float(np.sum(env.machine_capacities[:, 0]))

        rollup = _compute_episode_metric_rollup(
            assignment_records=assignments,
            total_tasks=total_tasks,
            unscheduled_tasks=unscheduled_tasks,
            cpu_capacity_total=cpu_capacity_total,
            episode_start_time=episode_start,
            episode_end_time=episode_end,
        )
        rollup.update(
            {
                "episode_id": int(episode_id),
                "total_reward": float(total_reward),
                "steps": int(step_count),
                "feasible_count": int(feasible_count),
                "defer_count": int(defer_count),
                "invalid_count": int(invalid_count),
                "feasible_rate": float(feasible_count / max(step_count, 1)),
                "defer_rate": float(defer_count / max(step_count, 1)),
                "invalid_rate": float(invalid_count / max(step_count, 1)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "truncate_reason": None if truncate_reason is None else str(truncate_reason),
                "assignment_count_mismatch": int(assignment_count_mismatch),
                "assignments": [row.as_dict() for row in assignments],
            }
        )
        episodes.append(rollup)

    env.close()

    summary = _aggregate_episode_rollups(episodes)
    truncate_reason_counts: dict[str, int] = {}
    for row in episodes:
        if not bool(row.get("truncated", False)):
            continue
        reason = str(row.get("truncate_reason") or "unknown")
        truncate_reason_counts[reason] = int(truncate_reason_counts.get(reason, 0) + 1)
    summary.update(
        {
            "scheduler": "RL-Hybrid" if bool(use_hybrid_scheduler) else "RL",
            "mean_total_reward": float(np.mean([float(row["total_reward"]) for row in episodes])),
            "mean_feasible_rate": float(np.mean([float(row["feasible_rate"]) for row in episodes])),
            "mean_defer_rate": float(np.mean([float(row["defer_rate"]) for row in episodes])),
            "mean_invalid_rate": float(np.mean([float(row["invalid_rate"]) for row in episodes])),
            "terminated_episodes": int(sum(1 for row in episodes if bool(row["terminated"]))),
            "truncated_episodes": int(sum(1 for row in episodes if bool(row["truncated"]))),
            "truncate_reason_counts": truncate_reason_counts,
            "deadline_slack_factor": float(deadline_slack_factor),
            "machine_capacity_scale": float(machine_capacity_scale),
            "machine_pool_size": None if machine_pool_size is None else int(machine_pool_size),
            "hybrid_scheduler_enabled": bool(use_hybrid_scheduler),
            "hybrid_defer_wait_ratio_threshold": float(hybrid_defer_wait_ratio_threshold),
            "hybrid_high_utilization_threshold": float(hybrid_high_utilization_threshold),
        }
    )

    return {
        "scheduler": "RL-Hybrid" if bool(use_hybrid_scheduler) else "RL",
        "model_path": str(model_path),
        "deterministic": bool(deterministic),
        "selected_episode_ids": selected_episode_ids,
        "summary": summary,
        "episodes": episodes,
    }


def comparison_row_from_summary(
    *,
    scheduler: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scheduler": str(scheduler),
        "deadline_miss_ratio": float(summary["deadline_miss_ratio"]),
        "mean_waiting_time": None if summary["mean_waiting_time"] is None else float(summary["mean_waiting_time"]),
        "mean_turnaround_time": None
        if summary["mean_turnaround_time"] is None
        else float(summary["mean_turnaround_time"]),
        "cpu_utilization": float(summary["cpu_utilization"]),
        "total_tasks": int(summary["total_tasks"]),
        "scheduled_tasks": int(summary["scheduled_tasks"]),
        "unscheduled_tasks": int(summary["unscheduled_tasks"]),
    }
