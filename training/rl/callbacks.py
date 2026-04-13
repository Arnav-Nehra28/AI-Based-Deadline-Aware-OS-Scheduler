from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError as exc:
    BaseCallback = object  # type: ignore[assignment,misc]
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None


class SchedulerMetricsCallback(BaseCallback):
    """Log scheduler-specific rollout metrics from `info` dictionaries."""

    def __init__(self, *, window_size: int = 1024, verbose: int = 0) -> None:
        if _SB3_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "stable-baselines3 is required for SchedulerMetricsCallback."
            ) from _SB3_IMPORT_ERROR

        super().__init__(verbose=verbose)  # type: ignore[misc]
        self.window_size = max(1, int(window_size))
        self.feasible_window: deque[float] = deque(maxlen=self.window_size)
        self.defer_window: deque[float] = deque(maxlen=self.window_size)
        self.invalid_window: deque[float] = deque(maxlen=self.window_size)
        self.component_windows: dict[str, deque[float]] = {}

    def _on_step(self) -> bool:
        infos: Any = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]
        if not isinstance(infos, list):
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue

            reward_components = info.get("reward_components")
            if not isinstance(reward_components, dict):
                reward_components = {}

            for component_name, component_value in reward_components.items():
                if not isinstance(component_value, (int, float, np.floating)):
                    continue
                if component_name not in self.component_windows:
                    self.component_windows[component_name] = deque(maxlen=self.window_size)
                self.component_windows[component_name].append(float(component_value))

            was_feasible = bool(info.get("was_feasible", False))
            was_defer = "defer_penalty" in reward_components
            was_invalid = "invalid_action_penalty" in reward_components or "overload_penalty" in reward_components

            self.feasible_window.append(1.0 if was_feasible else 0.0)
            self.defer_window.append(1.0 if was_defer else 0.0)
            self.invalid_window.append(1.0 if was_invalid else 0.0)

        return True

    def _on_rollout_end(self) -> None:
        if self.feasible_window:
            self.logger.record("scheduler/feasible_rate", float(np.mean(self.feasible_window)))
        if self.defer_window:
            self.logger.record("scheduler/defer_rate", float(np.mean(self.defer_window)))
        if self.invalid_window:
            self.logger.record("scheduler/invalid_rate", float(np.mean(self.invalid_window)))

        for component_name, values in sorted(self.component_windows.items()):
            if values:
                metric_name = f"scheduler/reward_component_{component_name}"
                self.logger.record(metric_name, float(np.mean(values)))

