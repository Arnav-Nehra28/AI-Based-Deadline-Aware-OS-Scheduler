from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None


def _parse_action_mask(info: dict[str, Any], action_count: int) -> np.ndarray:
    raw_mask = info.get("action_mask")
    if raw_mask is None:
        raw_mask = info.get("decision_action_mask")
    if raw_mask is None:
        return np.ones(action_count, dtype=bool)

    mask = np.asarray(raw_mask, dtype=np.int8).astype(bool, copy=False)
    if mask.shape != (action_count,):
        raise ValueError(
            f"Invalid action mask shape {mask.shape}; expected {(action_count,)}."
        )
    return mask


if gym is not None:

    class ActionMaskInfoWrapper(gym.Wrapper):
        """Expose action masks in the format expected by sb3-contrib MaskablePPO."""

        def __init__(self, env: gym.Env) -> None:
            super().__init__(env)
            if not hasattr(self.action_space, "n"):
                raise TypeError("ActionMaskInfoWrapper requires a discrete action space.")
            self._action_count = int(self.action_space.n)
            self._latest_action_mask = np.ones(self._action_count, dtype=bool)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            observation, info = self.env.reset(seed=seed, options=options)
            self._latest_action_mask = _parse_action_mask(info, self._action_count)
            return observation, info

        def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._latest_action_mask = _parse_action_mask(info, self._action_count)
            return observation, reward, terminated, truncated, info

        def action_masks(self) -> np.ndarray:
            return self._latest_action_mask.copy()

else:

    class ActionMaskInfoWrapper:
        """Fallback wrapper for environments when gymnasium is unavailable."""

        def __init__(self, env: Any) -> None:
            self.env = env
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            if not hasattr(self.action_space, "n"):
                raise TypeError("ActionMaskInfoWrapper requires a discrete action space.")
            self._action_count = int(self.action_space.n)
            self._latest_action_mask = np.ones(self._action_count, dtype=bool)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            observation, info = self.env.reset(seed=seed, options=options)
            self._latest_action_mask = _parse_action_mask(info, self._action_count)
            return observation, info

        def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._latest_action_mask = _parse_action_mask(info, self._action_count)
            return observation, reward, terminated, truncated, info

        def action_masks(self) -> np.ndarray:
            return self._latest_action_mask.copy()

        def close(self) -> None:
            close_fn = getattr(self.env, "close", None)
            if callable(close_fn):
                close_fn()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)

