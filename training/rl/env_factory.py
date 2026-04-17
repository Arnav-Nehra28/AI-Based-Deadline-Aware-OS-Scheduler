from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from rl_pipeline.env_dataset import RLEnvDataset, load_env_dataset
from rl_pipeline.environment import RewardWeights, TaskSchedulingEnv

try:
    from data_preprocessing.pipeline_config import RL_ENV_DATASET_JSON_GZ
except ImportError:
    RL_ENV_DATASET_JSON_GZ = Path("data/interim/rl_env_dataset.json.gz")


DEFAULT_RL_DATASET_PATH = Path(RL_ENV_DATASET_JSON_GZ)


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    """Resolve the RL dataset path, defaulting to pipeline config."""
    return DEFAULT_RL_DATASET_PATH if dataset_path is None else Path(dataset_path)


def load_scheduler_dataset(dataset_path: str | Path | None = None) -> RLEnvDataset:
    """Load the serialized scheduler environment dataset artifact."""
    return load_env_dataset(resolve_dataset_path(dataset_path))


def split_episode_ids(
    dataset: RLEnvDataset,
    *,
    eval_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split episode ids into train/eval buckets with deterministic shuffling."""
    episode_ids = sorted(dataset.episodes["episode_id"].astype(int).unique().tolist())
    if not episode_ids:
        raise ValueError("Dataset has no episode ids to split.")

    if len(episode_ids) == 1:
        # Small-dataset fallback: keep training/eval both available and deterministic.
        return episode_ids.copy(), episode_ids.copy()

    fraction = float(eval_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"eval_fraction must be between 0 and 1 (exclusive), received {eval_fraction}.")

    shuffled = np.asarray(episode_ids, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(shuffled)

    eval_count = int(round(len(shuffled) * fraction))
    eval_count = max(1, min(eval_count, len(shuffled) - 1))

    eval_ids = sorted(shuffled[:eval_count].tolist())
    train_ids = sorted(shuffled[eval_count:].tolist())
    return train_ids, eval_ids


def subset_dataset_by_episode_ids(dataset: RLEnvDataset, episode_ids: list[int]) -> RLEnvDataset:
    """Create a filtered view of the dataset containing only selected episodes."""
    selected_ids = sorted({int(episode_id) for episode_id in episode_ids})
    if not selected_ids:
        raise ValueError("Cannot build a dataset subset with an empty episode id list.")

    episodes = (
        dataset.episodes.loc[dataset.episodes["episode_id"].astype(int).isin(selected_ids)]
        .copy()
        .sort_values("episode_id")
        .reset_index(drop=True)
    )
    tasks = (
        dataset.tasks.loc[dataset.tasks["episode_id"].astype(int).isin(selected_ids)]
        .copy()
        .sort_values(["episode_id", "task_index"])
        .reset_index(drop=True)
    )
    if episodes.empty or tasks.empty:
        raise ValueError(
            "Filtered dataset is empty. Ensure the provided episode ids exist in the source artifact."
        )

    metadata = dict(dataset.metadata or {})
    metadata["episode_ids"] = selected_ids
    metadata["episode_count"] = len(episodes)

    return RLEnvDataset(
        tasks=tasks,
        machines=dataset.machines.copy().reset_index(drop=True),
        episodes=episodes,
        metadata=metadata,
    )


def build_scheduler_env(
    *,
    dataset: RLEnvDataset,
    top_k_candidates: int = 16,
    max_steps: int = 500,
    max_consecutive_defers: int = 30,
    invalid_action_limit: int = 30,
    machine_capacity_scale: float = 1.0,
    machine_pool_size: int | None = None,
    deadline_slack_factor: float = 2.0,
    reward_weights: RewardWeights | None = None,
    random_state: int = 42,
    randomize_on_reset: bool = True,
) -> TaskSchedulingEnv:
    """Build a scheduler environment instance from an in-memory dataset."""
    return TaskSchedulingEnv(
        dataset=dataset,
        top_k_candidates=top_k_candidates,
        max_steps=max_steps,
        max_consecutive_defers=max_consecutive_defers,
        invalid_action_limit=invalid_action_limit,
        machine_capacity_scale=machine_capacity_scale,
        machine_pool_size=machine_pool_size,
        deadline_slack_factor=deadline_slack_factor,
        reward_weights=reward_weights,
        random_state=random_state,
        randomize_on_reset=randomize_on_reset,
    )


def make_env_fn(
    *,
    dataset: RLEnvDataset,
    top_k_candidates: int = 16,
    max_steps: int = 500,
    max_consecutive_defers: int = 30,
    invalid_action_limit: int = 30,
    machine_capacity_scale: float = 1.0,
    machine_pool_size: int | None = None,
    deadline_slack_factor: float = 2.0,
    reward_weights: RewardWeights | None = None,
    random_state: int = 42,
    randomize_on_reset: bool = True,
    wrapper_fn: Callable[[TaskSchedulingEnv], object] | None = None,
) -> Callable[[], object]:
    """Return a thunk for vectorized environment constructors (DummyVecEnv/SubprocVecEnv)."""

    def _init() -> object:
        env = build_scheduler_env(
            dataset=dataset,
            top_k_candidates=top_k_candidates,
            max_steps=max_steps,
            max_consecutive_defers=max_consecutive_defers,
            invalid_action_limit=invalid_action_limit,
            machine_capacity_scale=machine_capacity_scale,
            machine_pool_size=machine_pool_size,
            deadline_slack_factor=deadline_slack_factor,
            reward_weights=reward_weights,
            random_state=random_state,
            randomize_on_reset=randomize_on_reset,
        )
        return wrapper_fn(env) if wrapper_fn is not None else env

    return _init
