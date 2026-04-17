from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HybridSchedulerConfig:
    defer_wait_ratio_threshold: float = 2.0
    high_utilization_threshold: float = 0.90
    min_dynamic_defer_wait_ratio_threshold: float = 0.75
    queue_pressure_weight: float = 0.80
    near_deadline_weight: float = 1.20
    prefer_heuristic_when_feasible: bool = True


class HybridScheduler:
    """
    RL + heuristic fallback scheduler.

    Decision rules:
    1. Query RL policy first.
    2. If RL defers but task has waited too long and a feasible machine exists:
       force-assign via best-fit fallback.
    3. If RL picks a feasible machine that would create extreme post-utilization:
       switch to balanced fallback.
    """

    def __init__(self, config: HybridSchedulerConfig | None = None) -> None:
        self.config = config or HybridSchedulerConfig()

    def select_action(
        self,
        *,
        model: Any,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        env: Any,
        deterministic: bool,
    ) -> int:
        action, _ = model.predict(
            observation,
            deterministic=bool(deterministic),
            action_masks=action_mask,
        )
        action_value = int(np.asarray(action).item())
        defer_action = int(getattr(env, "defer_action"))

        task_row = env._current_task()  # internal env API used intentionally for fallback logic
        if task_row is None:
            return action_value

        feasible_details = [
            detail
            for detail in list(getattr(env, "_candidate_details", []))
            if bool(detail.get("is_feasible", False))
        ]
        if not feasible_details:
            return action_value

        wait_ratio = self._task_wait_ratio(task_row=task_row, current_time=float(env.current_time))
        queue_pressure = float(len(getattr(env, "pending_queue", [])) / max(getattr(env, "max_queue_size", 1), 1))
        near_deadline_fraction = float(self._fraction_tasks_near_deadline(env=env))
        dynamic_wait_threshold = max(
            float(self.config.min_dynamic_defer_wait_ratio_threshold),
            float(self.config.defer_wait_ratio_threshold)
            - float(self.config.queue_pressure_weight) * queue_pressure
            - float(self.config.near_deadline_weight) * near_deadline_fraction,
        )
        deadline_aware_action = self._deadline_aware_action_index(
            env=env,
            task_row=task_row,
            feasible_details=feasible_details,
        )

        # Keep queue healthy: when we already have a feasible machine, prefer dispatch over defer chains.
        if bool(self.config.prefer_heuristic_when_feasible):
            return int(deadline_aware_action)

        if action_value == defer_action and wait_ratio > dynamic_wait_threshold:
            return int(deadline_aware_action)

        if action_value != defer_action and action_value < len(getattr(env, "_candidate_details", [])):
            selected_detail = env._candidate_details[action_value]
            if bool(selected_detail.get("is_feasible", False)):
                post_util = float(selected_detail.get("max_post_utilization", 0.0))
                if post_util > float(self.config.high_utilization_threshold):
                    return int(self._balanced_action_index(env=env, feasible_details=feasible_details))
                selected_lateness = self._predicted_lateness(env=env, task_row=task_row, detail=selected_detail)
                best_deadline_action = deadline_aware_action
                best_detail = env._candidate_details[best_deadline_action]
                best_lateness = self._predicted_lateness(env=env, task_row=task_row, detail=best_detail)
                if selected_lateness > 0.0 and best_lateness + 1e-6 < selected_lateness:
                    return int(best_deadline_action)

        return action_value

    @staticmethod
    def _task_wait_ratio(*, task_row: dict[str, Any], current_time: float) -> float:
        wait_time = max(0.0, float(current_time) - float(task_row["arrival_time"]))
        duration = max(float(task_row["duration"]), 1.0)
        return float(wait_time / duration)

    @staticmethod
    def _fraction_tasks_near_deadline(*, env: Any) -> float:
        pending = list(getattr(env, "pending_queue", []))
        if not pending:
            return 0.0
        current_episode_tasks = getattr(env, "current_episode_tasks")
        current_time = float(getattr(env, "current_time"))
        deadline_slack_factor = float(getattr(getattr(env, "config", object()), "deadline_slack_factor", 2.0))
        near = 0
        for task_position in pending:
            task = current_episode_tasks.iloc[int(task_position)]
            duration = max(float(task["duration"]), 1.0)
            deadline = float(task["arrival_time"]) + deadline_slack_factor * duration
            urgency = np.clip(1.0 - (deadline - current_time) / duration, 0.0, 2.0)
            if urgency >= 0.8:
                near += 1
        return float(near / len(pending))

    @staticmethod
    def _predicted_lateness(*, env: Any, task_row: dict[str, Any], detail: dict[str, Any]) -> float:
        current_time = float(getattr(env, "current_time"))
        duration = max(float(task_row["duration"]), 1.0)
        deadline_slack_factor = float(getattr(getattr(env, "config", object()), "deadline_slack_factor", 2.0))
        deadline = float(task_row["arrival_time"]) + deadline_slack_factor * duration
        completion_time = current_time + duration
        return float(max(0.0, completion_time - deadline))

    @staticmethod
    def _deadline_aware_action_index(*, env: Any, task_row: dict[str, Any], feasible_details: list[dict[str, Any]]) -> int:
        best_detail = min(
            feasible_details,
            key=lambda detail: (
                HybridScheduler._predicted_lateness(env=env, task_row=task_row, detail=detail),
                -float(detail.get("fit_score", 0.0)),
                -float(detail.get("balance_score", 0.0)),
                -float(detail.get("fragmentation_score", 0.0)),
            ),
        )
        return int(env._candidate_details.index(best_detail))

    @staticmethod
    def _best_fit_action_index(*, env: Any, feasible_details: list[dict[str, Any]]) -> int:
        best_detail = max(
            feasible_details,
            key=lambda detail: (
                float(detail.get("fit_score", 0.0)),
                float(detail.get("balance_score", 0.0)),
                float(detail.get("fragmentation_score", 0.0)),
            ),
        )
        return int(env._candidate_details.index(best_detail))

    @staticmethod
    def _balanced_action_index(*, env: Any, feasible_details: list[dict[str, Any]]) -> int:
        best_detail = max(
            feasible_details,
            key=lambda detail: (
                float(detail.get("balance_score", 0.0)),
                float(detail.get("fragmentation_score", 0.0)),
                -float(detail.get("max_post_utilization", 1.0)),
            ),
        )
        return int(env._candidate_details.index(best_detail))
