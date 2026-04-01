from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from pathlib import Path
from typing import Any

import numpy as np

from .env_dataset import RLEnvDataset, load_env_dataset
from .gym_compat import Env, spaces


@dataclass(frozen=True)
class RewardWeights:
    feasible_bonus: float = 2.5
    overload_penalty: float = -4.0
    invalid_action_penalty: float = -5.0
    defer_penalty: float = -1.0
    wait_penalty_weight: float = 0.50
    balance_bonus_weight: float = 0.75
    fragmentation_penalty_weight: float = 0.35
    hotspot_penalty_weight: float = 0.35
    historical_match_bonus: float = 0.10


@dataclass(frozen=True)
class SchedulerEnvConfig:
    top_k_candidates: int = 16
    max_steps: int = 160
    max_consecutive_defers: int = 8
    invalid_action_limit: int = 8
    randomize_on_reset: bool = True


@dataclass
class StepInfo:
    episode_id: int
    task_id: str | None
    task_index: int | None
    decision_time: float
    current_time: float
    selected_machine_id: str | None
    candidate_machine_ids: list[str | None]
    action_mask: np.ndarray
    reward_components: dict[str, float] = field(default_factory=dict)
    was_feasible: bool = False
    historical_machine_id: str | None = None
    pending_queue_size: int = 0
    running_jobs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "task_index": self.task_index,
            "decision_time": self.decision_time,
            "current_time": self.current_time,
            "selected_machine_id": self.selected_machine_id,
            "candidate_machine_ids": list(self.candidate_machine_ids),
            "action_mask": self.action_mask.astype(np.int8).copy(),
            "reward_components": dict(self.reward_components),
            "was_feasible": bool(self.was_feasible),
            "historical_machine_id": self.historical_machine_id,
            "pending_queue_size": int(self.pending_queue_size),
            "running_jobs": int(self.running_jobs),
        }


class TaskSchedulingEnv(Env):
    """Gymnasium-compatible online scheduler simulator with top-k masked actions."""

    metadata = {"render_modes": []}

    TASK_FEATURE_DIM = 8
    CANDIDATE_FEATURE_DIM = 13
    FLEET_SUMMARY_DIM = 12

    def __init__(
        self,
        dataset: RLEnvDataset | None = None,
        dataset_path: str | Path | None = None,
        *,
        top_k_candidates: int = 16,
        max_steps: int = 160,
        max_consecutive_defers: int = 8,
        invalid_action_limit: int = 8,
        reward_weights: RewardWeights | None = None,
        random_state: int = 42,
        randomize_on_reset: bool = True,
    ) -> None:
        super().__init__()

        if dataset is None and dataset_path is None:
            raise ValueError("Provide either dataset or dataset_path when creating TaskSchedulingEnv.")
        if dataset is not None and dataset_path is not None:
            raise ValueError("Pass only one of dataset or dataset_path, not both.")

        self.dataset = dataset if dataset is not None else load_env_dataset(dataset_path)  # type: ignore[arg-type]
        self.config = SchedulerEnvConfig(
            top_k_candidates=int(top_k_candidates),
            max_steps=int(max_steps),
            max_consecutive_defers=int(max_consecutive_defers),
            invalid_action_limit=int(invalid_action_limit),
            randomize_on_reset=bool(randomize_on_reset),
        )
        self.reward_weights = reward_weights or RewardWeights()
        self.defer_action = self.config.top_k_candidates
        self._rng = np.random.default_rng(random_state)
        self._episode_cursor = 0

        self.machines = self.dataset.machines.copy().reset_index(drop=True)
        self.tasks = self.dataset.tasks.copy().reset_index(drop=True)
        self.episodes = self.dataset.episodes.copy().sort_values("episode_id").reset_index(drop=True)

        self.machines["machine_id"] = self.machines["machine_id"].astype(str)
        self.tasks["historical_machine_id"] = self.tasks["historical_machine_id"].astype(str)
        self.tasks["task_id"] = self.tasks["task_id"].astype(str)

        self.machine_ids = self.machines["machine_id"].tolist()
        self.machine_to_index = {machine_id: index for index, machine_id in enumerate(self.machine_ids)}
        self.machine_capacities = self.machines[["cpu_capacity", "mem_capacity", "disk_capacity"]].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        self.machine_capacities_safe = np.where(self.machine_capacities > 0, self.machine_capacities, 1.0)

        self.max_capacity_by_dim = np.maximum(self.machine_capacities_safe.max(axis=0), 1.0)
        self.max_duration = max(float(self.tasks["duration"].max()), 1.0)
        self.max_arrival_time = max(float(self.tasks["arrival_time"].max()), 1.0)
        self.max_queue_size = max(int(self.episodes["task_count"].max()), 1)

        self._episode_tables = {
            int(episode_id): frame.sort_values("task_index").reset_index(drop=True)
            for episode_id, frame in self.tasks.groupby("episode_id", sort=True)
        }
        self._episode_ids = self.episodes["episode_id"].astype(int).tolist()

        self.action_space = spaces.Discrete(self.config.top_k_candidates + 1)
        self.observation_space = spaces.Dict(
            {
                "task_features": spaces.Box(
                    low=np.zeros(self.TASK_FEATURE_DIM, dtype=np.float32),
                    high=np.full(self.TASK_FEATURE_DIM, np.inf, dtype=np.float32),
                    shape=(self.TASK_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                "candidate_features": spaces.Box(
                    low=np.full((self.config.top_k_candidates, self.CANDIDATE_FEATURE_DIM), -np.inf, dtype=np.float32),
                    high=np.full((self.config.top_k_candidates, self.CANDIDATE_FEATURE_DIM), np.inf, dtype=np.float32),
                    shape=(self.config.top_k_candidates, self.CANDIDATE_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "fleet_summary": spaces.Box(
                    low=np.full(self.FLEET_SUMMARY_DIM, -np.inf, dtype=np.float32),
                    high=np.full(self.FLEET_SUMMARY_DIM, np.inf, dtype=np.float32),
                    shape=(self.FLEET_SUMMARY_DIM,),
                    dtype=np.float32,
                ),
            }
        )

        self.current_episode_id: int | None = None
        self.current_time = 0.0
        self.step_count = 0
        self.consecutive_defers = 0
        self.invalid_action_count = 0
        self.defer_count = 0
        self.arrival_cursor = 0
        self.pending_queue: list[int] = []
        self.running_jobs: list[tuple[float, int, int, np.ndarray]] = []
        self.machine_residual = self.machine_capacities.copy()
        self.current_episode_tasks = self._episode_tables[self._episode_ids[0]].copy()
        self._job_counter = 0

        self._candidate_machine_indices: list[int] = []
        self._candidate_machine_ids: list[str | None] = [None] * self.config.top_k_candidates
        self._action_mask = np.zeros(self.config.top_k_candidates + 1, dtype=np.int8)
        self._candidate_features = np.zeros(
            (self.config.top_k_candidates, self.CANDIDATE_FEATURE_DIM), dtype=np.float32
        )
        self._candidate_details: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        episode_id = self._select_episode_id(options=options or {})
        self.current_episode_id = episode_id
        self.current_episode_tasks = self._episode_tables[episode_id].copy()

        self.current_time = float(self.current_episode_tasks["arrival_time"].min())
        self.step_count = 0
        self.consecutive_defers = 0
        self.invalid_action_count = 0
        self.defer_count = 0
        self.arrival_cursor = 0
        self.pending_queue = []
        self.running_jobs = []
        self.machine_residual = self.machine_capacities.copy()
        self._job_counter = 0

        self._advance_until_decision()
        self._refresh_decision_state()
        observation = self._build_observation()
        info = self._build_info(
            decision_task=self._current_task(),
            decision_time=self.current_time,
            decision_candidate_machine_ids=list(self._candidate_machine_ids),
            decision_action_mask=self._action_mask.copy(),
            selected_machine_id=None,
            reward_components={},
            was_feasible=False,
        )
        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.current_episode_id is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._current_task() is None:
            raise RuntimeError("Episode is already done. Call reset() before the next step.")

        decision_task = self._current_task()
        assert decision_task is not None
        decision_time = self.current_time
        decision_candidate_machine_ids = list(self._candidate_machine_ids)
        decision_action_mask = self._action_mask.copy()
        reward_components: dict[str, float]
        selected_machine_id: str | None = None
        was_feasible = False

        action_value = int(action)
        self.step_count += 1

        if action_value == self.defer_action:
            reward_components = self._apply_defer()
        elif not self.action_space.contains(action_value) or self._action_mask[action_value] == 0:
            reward_components = self._apply_invalid_action()
        else:
            candidate_detail = self._candidate_details[action_value]
            selected_machine_id = str(candidate_detail["machine_id"])
            if bool(candidate_detail["is_feasible"]):
                reward_components = self._apply_feasible_assignment(
                    machine_index=int(candidate_detail["machine_index"]),
                    task_row=decision_task,
                    candidate_detail=candidate_detail,
                )
                was_feasible = True
            else:
                reward_components = self._apply_infeasible_assignment(candidate_detail=candidate_detail, task_row=decision_task)

        terminated = self._is_terminated()
        truncated = (
            self.step_count >= self.config.max_steps
            or self.consecutive_defers >= self.config.max_consecutive_defers
            or self.invalid_action_count >= self.config.invalid_action_limit
        )

        if not terminated and not truncated:
            self._advance_until_decision()
        self._refresh_decision_state()

        observation = self._build_observation()
        info = self._build_info(
            decision_task=decision_task,
            decision_time=decision_time,
            decision_candidate_machine_ids=decision_candidate_machine_ids,
            decision_action_mask=decision_action_mask,
            selected_machine_id=selected_machine_id,
            reward_components=reward_components,
            was_feasible=was_feasible,
        )
        reward = float(sum(reward_components.values()))
        return observation, reward, terminated, truncated, info

    def render(self) -> dict[str, Any]:
        return {
            "episode_id": self.current_episode_id,
            "current_time": self.current_time,
            "pending_queue_size": len(self.pending_queue),
            "running_jobs": len(self.running_jobs),
        }

    def close(self) -> None:
        return None

    def _select_episode_id(self, options: dict[str, Any]) -> int:
        requested_episode_id = options.get("episode_id")
        if requested_episode_id is not None:
            episode_id = int(requested_episode_id)
            if episode_id not in self._episode_tables:
                raise ValueError(f"Unknown episode_id requested at reset(): {episode_id}")
            return episode_id

        if self.config.randomize_on_reset:
            return int(self._rng.choice(self._episode_ids))

        episode_id = int(self._episode_ids[self._episode_cursor % len(self._episode_ids)])
        self._episode_cursor += 1
        return episode_id

    def _current_task(self) -> dict[str, Any] | None:
        if not self.pending_queue:
            return None
        task_position = self.pending_queue[0]
        row = self.current_episode_tasks.iloc[task_position]
        return row.to_dict()

    def _advance_until_decision(self) -> None:
        while True:
            self._release_finished_jobs(up_to=self.current_time)
            self._enqueue_arrived_tasks(up_to=self.current_time)
            if self.pending_queue:
                return
            if self.arrival_cursor >= len(self.current_episode_tasks):
                return
            self.current_time = max(
                self.current_time,
                float(self.current_episode_tasks.iloc[self.arrival_cursor]["arrival_time"]),
            )

    def _release_finished_jobs(self, up_to: float) -> None:
        while self.running_jobs and self.running_jobs[0][0] <= up_to:
            _, _, machine_index, demand = heapq.heappop(self.running_jobs)
            self.machine_residual[machine_index] = np.minimum(
                self.machine_capacities[machine_index],
                self.machine_residual[machine_index] + demand,
            )

    def _enqueue_arrived_tasks(self, up_to: float) -> None:
        while self.arrival_cursor < len(self.current_episode_tasks):
            row = self.current_episode_tasks.iloc[self.arrival_cursor]
            if float(row["arrival_time"]) > up_to:
                break
            self.pending_queue.append(self.arrival_cursor)
            self.arrival_cursor += 1

    def _refresh_decision_state(self) -> None:
        self._candidate_machine_indices = []
        self._candidate_machine_ids = [None] * self.config.top_k_candidates
        self._action_mask = np.zeros(self.config.top_k_candidates + 1, dtype=np.int8)
        self._candidate_features = np.zeros(
            (self.config.top_k_candidates, self.CANDIDATE_FEATURE_DIM), dtype=np.float32
        )
        self._candidate_details = []

        task_row = self._current_task()
        if task_row is None:
            return

        shortlist = self._rank_machine_candidates(task_row=task_row)
        for slot, detail in enumerate(shortlist):
            self._candidate_machine_indices.append(int(detail["machine_index"]))
            self._candidate_machine_ids[slot] = str(detail["machine_id"])
            self._candidate_features[slot] = np.asarray(detail["feature_row"], dtype=np.float32)
            self._candidate_details.append(detail)
            self._action_mask[slot] = 1

        self._action_mask[self.defer_action] = 1

    def _rank_machine_candidates(self, task_row: dict[str, Any]) -> list[dict[str, Any]]:
        demand = np.array(
            [task_row["cpu_demand"], task_row["mem_demand"], task_row["disk_demand"]],
            dtype=np.float32,
        )
        residual = self.machine_residual
        post_residual = residual - demand
        residual_ratio = residual / self.machine_capacities_safe
        post_ratio = post_residual / self.machine_capacities_safe
        overload = np.clip(-post_residual, 0.0, None) / self.machine_capacities_safe
        overload_amount = overload.sum(axis=1)
        feasible_mask = np.all(post_residual >= -1e-6, axis=1)

        post_utilization = 1.0 - np.clip(post_residual, 0.0, None) / self.machine_capacities_safe
        fit_score = 1.0 - np.mean(np.clip(post_ratio, 0.0, 1.0), axis=1)
        balance_score = 1.0 - np.std(np.clip(post_utilization, 0.0, 1.5), axis=1)
        fragmentation_score = 1.0 - (
            np.max(np.clip(post_ratio, 0.0, 1.0), axis=1) - np.min(np.clip(post_ratio, 0.0, 1.0), axis=1)
        )
        overall_score = np.where(
            feasible_mask,
            1.5 * fit_score + 1.0 * balance_score + 0.75 * fragmentation_score,
            -10.0 - overload_amount,
        )

        feasible_indices = np.where(feasible_mask)[0]
        infeasible_indices = np.where(~feasible_mask)[0]

        ordered_feasible = feasible_indices[np.argsort(overall_score[feasible_indices])[::-1]].tolist()
        ordered_infeasible = infeasible_indices[np.argsort(overload_amount[infeasible_indices])].tolist()
        shortlist_indices = (ordered_feasible + ordered_infeasible)[: self.config.top_k_candidates]

        historical_machine_id = str(task_row["historical_machine_id"])
        details: list[dict[str, Any]] = []
        for machine_index in shortlist_indices:
            machine_id = self.machine_ids[machine_index]
            details.append(
                {
                    "machine_index": int(machine_index),
                    "machine_id": machine_id,
                    "is_feasible": bool(feasible_mask[machine_index]),
                    "overload_amount": float(overload_amount[machine_index]),
                    "fit_score": float(fit_score[machine_index]),
                    "balance_score": float(balance_score[machine_index]),
                    "fragmentation_score": float(fragmentation_score[machine_index]),
                    "max_post_utilization": float(np.max(post_utilization[machine_index])),
                    "historical_match": bool(machine_id == historical_machine_id),
                    "feature_row": np.array(
                        [
                            1.0,
                            float(feasible_mask[machine_index]),
                            float(residual_ratio[machine_index][0]),
                            float(residual_ratio[machine_index][1]),
                            float(residual_ratio[machine_index][2]),
                            float(post_ratio[machine_index][0]),
                            float(post_ratio[machine_index][1]),
                            float(post_ratio[machine_index][2]),
                            float(overload_amount[machine_index]),
                            float(fit_score[machine_index]),
                            float(balance_score[machine_index]),
                            float(fragmentation_score[machine_index]),
                            float(machine_id == historical_machine_id),
                        ],
                        dtype=np.float32,
                    ),
                }
            )
        return details

    def _build_observation(self) -> dict[str, np.ndarray]:
        task_row = self._current_task()
        if task_row is None:
            return {
                "task_features": np.zeros(self.TASK_FEATURE_DIM, dtype=np.float32),
                "candidate_features": self._candidate_features.copy(),
                "fleet_summary": np.zeros(self.FLEET_SUMMARY_DIM, dtype=np.float32),
            }

        demand = np.array(
            [task_row["cpu_demand"], task_row["mem_demand"], task_row["disk_demand"]],
            dtype=np.float32,
        )
        wait_time = max(0.0, float(self.current_time - float(task_row["arrival_time"])))
        resource_pressure = float(np.mean(demand / self.max_capacity_by_dim))
        task_features = np.array(
            [
                float(demand[0] / self.max_capacity_by_dim[0]),
                float(demand[1] / self.max_capacity_by_dim[1]),
                float(demand[2] / self.max_capacity_by_dim[2]),
                float(task_row["duration"] / self.max_duration),
                float(wait_time / self.max_duration),
                float(task_row["arrival_time"] / self.max_arrival_time),
                float(len(self.pending_queue) / self.max_queue_size),
                resource_pressure,
            ],
            dtype=np.float32,
        )

        utilization = 1.0 - (self.machine_residual / self.machine_capacities_safe)
        fragmentation = np.std(np.clip(self.machine_residual / self.machine_capacities_safe, 0.0, 1.0), axis=1)
        real_candidate_count = max(1, len(self._candidate_details))
        feasible_fraction = sum(1 for detail in self._candidate_details if detail["is_feasible"]) / real_candidate_count

        fleet_summary = np.array(
            [
                float(self.current_time / self.max_arrival_time),
                float(len(self.pending_queue) / self.max_queue_size),
                float(len(self.running_jobs) / self.max_queue_size),
                float(feasible_fraction),
                float(np.mean(utilization[:, 0])),
                float(np.mean(utilization[:, 1])),
                float(np.mean(utilization[:, 2])),
                float(np.max(utilization[:, 0])),
                float(np.max(utilization[:, 1])),
                float(np.max(utilization[:, 2])),
                float(np.mean(fragmentation)),
                float(self.consecutive_defers / max(self.config.max_consecutive_defers, 1)),
            ],
            dtype=np.float32,
        )

        return {
            "task_features": task_features,
            "candidate_features": self._candidate_features.copy(),
            "fleet_summary": fleet_summary,
        }

    def _apply_defer(self) -> dict[str, float]:
        if self.pending_queue:
            current_task = self.pending_queue.pop(0)
            self.pending_queue.append(current_task)

        self.defer_count += 1
        self.consecutive_defers += 1

        decision_task = self._current_task()
        wait_time = 0.0 if decision_task is None else max(0.0, self.current_time - float(decision_task["arrival_time"]))
        reward_components = {
            "defer_penalty": self.reward_weights.defer_penalty,
            "wait_penalty": -self.reward_weights.wait_penalty_weight * float(wait_time / self.max_duration),
        }
        next_event_time = self._next_external_event_time()
        if len(self.pending_queue) <= 1 and next_event_time is not None and next_event_time > self.current_time:
            self._advance_clock(next_event_time)
        return reward_components

    def _apply_invalid_action(self) -> dict[str, float]:
        self.invalid_action_count += 1
        self.consecutive_defers = 0
        reward_components = {"invalid_action_penalty": self.reward_weights.invalid_action_penalty}
        return reward_components

    def _apply_infeasible_assignment(
        self,
        candidate_detail: dict[str, Any],
        task_row: dict[str, Any],
    ) -> dict[str, float]:
        self.invalid_action_count += 1
        self.consecutive_defers = 0
        wait_time = max(0.0, float(self.current_time - float(task_row["arrival_time"])))
        reward_components = {
            "overload_penalty": self.reward_weights.overload_penalty - float(candidate_detail["overload_amount"]),
            "wait_penalty": -self.reward_weights.wait_penalty_weight * float(wait_time / self.max_duration),
        }
        return reward_components

    def _apply_feasible_assignment(
        self,
        machine_index: int,
        task_row: dict[str, Any],
        candidate_detail: dict[str, Any],
    ) -> dict[str, float]:
        current_task_position = self.pending_queue.pop(0)
        assert int(task_row["task_index"]) == int(self.current_episode_tasks.iloc[current_task_position]["task_index"])

        demand = np.array(
            [task_row["cpu_demand"], task_row["mem_demand"], task_row["disk_demand"]],
            dtype=np.float32,
        )
        self.machine_residual[machine_index] -= demand
        self.machine_residual[machine_index] = np.clip(
            self.machine_residual[machine_index],
            a_min=0.0,
            a_max=self.machine_capacities[machine_index],
        )

        end_time = float(self.current_time + float(task_row["duration"]))
        heapq.heappush(self.running_jobs, (end_time, self._job_counter, machine_index, demand.copy()))
        self._job_counter += 1

        self.consecutive_defers = 0
        self.invalid_action_count = 0

        wait_time = max(0.0, float(self.current_time - float(task_row["arrival_time"])))
        fragmentation_penalty = self.reward_weights.fragmentation_penalty_weight * max(
            0.0, 1.0 - float(candidate_detail["fragmentation_score"])
        )
        hotspot_penalty = self.reward_weights.hotspot_penalty_weight * max(
            0.0, float(candidate_detail["max_post_utilization"]) - 1.0
        )
        reward_components = {
            "feasible_bonus": self.reward_weights.feasible_bonus,
            "wait_penalty": -self.reward_weights.wait_penalty_weight * float(wait_time / self.max_duration),
            "balance_bonus": self.reward_weights.balance_bonus_weight * max(
                0.0, float(candidate_detail["balance_score"])
            ),
            "fragmentation_penalty": -fragmentation_penalty,
            "hotspot_penalty": -hotspot_penalty,
            "historical_bonus": self.reward_weights.historical_match_bonus
            if bool(candidate_detail["historical_match"])
            else 0.0,
        }
        return reward_components

    def _next_external_event_time(self) -> float | None:
        next_times: list[float] = []
        if self.arrival_cursor < len(self.current_episode_tasks):
            next_times.append(float(self.current_episode_tasks.iloc[self.arrival_cursor]["arrival_time"]))
        if self.running_jobs:
            next_times.append(float(self.running_jobs[0][0]))

        future_times = [candidate for candidate in next_times if candidate > self.current_time]
        if not future_times:
            return None
        return min(future_times)

    def _advance_clock(self, target_time: float) -> None:
        self.current_time = max(self.current_time, float(target_time))
        self._release_finished_jobs(up_to=self.current_time)
        self._enqueue_arrived_tasks(up_to=self.current_time)

    def _is_terminated(self) -> bool:
        return self.arrival_cursor >= len(self.current_episode_tasks) and not self.pending_queue

    def _build_info(
        self,
        *,
        decision_task: dict[str, Any] | None,
        decision_time: float,
        decision_candidate_machine_ids: list[str | None],
        decision_action_mask: np.ndarray,
        selected_machine_id: str | None,
        reward_components: dict[str, float],
        was_feasible: bool,
    ) -> dict[str, Any]:
        task_id = None if decision_task is None else str(decision_task["task_id"])
        task_index = None if decision_task is None else int(decision_task["task_index"])
        historical_machine_id = None if decision_task is None else str(decision_task["historical_machine_id"])
        info = StepInfo(
            episode_id=int(self.current_episode_id if self.current_episode_id is not None else -1),
            task_id=task_id,
            task_index=task_index,
            decision_time=float(decision_time),
            current_time=float(self.current_time),
            selected_machine_id=selected_machine_id,
            candidate_machine_ids=list(self._candidate_machine_ids),
            action_mask=self._action_mask.copy(),
            reward_components=reward_components,
            was_feasible=was_feasible,
            historical_machine_id=historical_machine_id,
            pending_queue_size=len(self.pending_queue),
            running_jobs=len(self.running_jobs),
        ).as_dict()
        info["decision_candidate_machine_ids"] = list(decision_candidate_machine_ids)
        info["decision_action_mask"] = decision_action_mask.astype(np.int8).copy()
        return info
