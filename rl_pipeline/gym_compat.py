from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    Env = gym.Env
    GYMNASIUM_AVAILABLE = True
except ModuleNotFoundError:
    GYMNASIUM_AVAILABLE = False

    class Env:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
            del options
            if seed is not None or not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng(seed)

    @dataclass
    class _Space:
        shape: tuple[int, ...] | None = None
        dtype: Any = np.float32

        def contains(self, x: Any) -> bool:
            return True

    class Box(_Space):
        def __init__(self, low: Any, high: Any, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
            super().__init__(shape=shape, dtype=dtype)
            self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape)
            self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape)

        def contains(self, x: Any) -> bool:
            array = np.asarray(x, dtype=self.dtype)
            if array.shape != self.shape:
                return False
            return bool(np.all(array >= self.low) and np.all(array <= self.high))

    class Discrete(_Space):
        def __init__(self, n: int) -> None:
            super().__init__(shape=(), dtype=np.int64)
            self.n = int(n)

        def contains(self, x: Any) -> bool:
            try:
                value = int(x)
            except (TypeError, ValueError):
                return False
            return 0 <= value < self.n

    class Dict(_Space):
        def __init__(self, spaces_dict: dict[str, _Space]) -> None:
            super().__init__(shape=None, dtype=None)
            self.spaces = spaces_dict

        def contains(self, x: Any) -> bool:
            if not isinstance(x, dict):
                return False
            if set(x) != set(self.spaces):
                return False
            return all(self.spaces[key].contains(x[key]) for key in self.spaces)

    class _SpacesModule:
        Box = Box
        Discrete = Discrete
        Dict = Dict

    spaces = _SpacesModule()
