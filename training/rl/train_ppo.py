from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .callbacks import SchedulerMetricsCallback
    from .env_factory import (
        DEFAULT_RL_DATASET_PATH,
        build_scheduler_env,
        load_scheduler_dataset,
        split_episode_ids,
        subset_dataset_by_episode_ids,
    )
    from .wrappers import ActionMaskInfoWrapper
except ImportError:
    from callbacks import SchedulerMetricsCallback
    from env_factory import (
        DEFAULT_RL_DATASET_PATH,
        build_scheduler_env,
        load_scheduler_dataset,
        split_episode_ids,
        subset_dataset_by_episode_ids,
    )
    from wrappers import ActionMaskInfoWrapper

from rl_pipeline.environment import RewardWeights
from rl_pipeline.env_dataset import RLEnvDataset


def _require_rl_dependencies() -> dict[str, Any]:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import get_schedule_fn
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing RL dependency for PPO training. Install with:\n"
            "pip install gymnasium stable-baselines3 sb3-contrib"
        ) from exc

    return {
        "MaskablePPO": MaskablePPO,
        "MaskableEvalCallback": MaskableEvalCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "DummyVecEnv": DummyVecEnv,
        "get_schedule_fn": get_schedule_fn,
        "Monitor": Monitor,
        "VecMonitor": VecMonitor,
    }


def _parse_hidden_dims(raw_value: str) -> list[int]:
    dims: list[int] = []
    for chunk in raw_value.split(","):
        value = chunk.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                f"Policy hidden dimensions must be positive integers. Received: {parsed}"
            )
        dims.append(parsed)
    if not dims:
        raise argparse.ArgumentTypeError(
            "Policy hidden dimensions cannot be empty. Example: --policy-hidden-dims 256,256"
        )
    return dims


def _build_run_dir(output_root: Path, run_name: str | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"ppo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    candidate = output_root / run_name
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{run_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _parse_episode_ids(raw_value: str | None) -> list[int]:
    if raw_value is None:
        return []
    values: list[int] = []
    for chunk in str(raw_value).split(","):
        token = chunk.strip()
        if token:
            values.append(int(token))
    return sorted(set(values))


def _resolve_episode_split(
    *,
    dataset: RLEnvDataset,
    seed: int,
    eval_fraction: float,
    train_episode_ids_arg: str | None,
    eval_episode_ids_arg: str | None,
) -> tuple[list[int], list[int]]:
    available_ids = sorted({int(value) for value in dataset.episodes["episode_id"].tolist()})
    available_set = set(available_ids)
    explicit_train = _parse_episode_ids(train_episode_ids_arg)
    explicit_eval = _parse_episode_ids(eval_episode_ids_arg)

    if not explicit_train and not explicit_eval:
        return split_episode_ids(
            dataset,
            eval_fraction=float(eval_fraction),
            seed=int(seed),
        )

    if not set(explicit_train).issubset(available_set):
        unknown = sorted(set(explicit_train) - available_set)
        raise ValueError(f"Unknown episode ids in --train-episode-ids: {unknown}")
    if not set(explicit_eval).issubset(available_set):
        unknown = sorted(set(explicit_eval) - available_set)
        raise ValueError(f"Unknown episode ids in --eval-episode-ids: {unknown}")

    train_set = set(explicit_train)
    eval_set = set(explicit_eval)
    if not train_set:
        train_set = available_set - eval_set
    if not eval_set:
        eval_set = available_set - train_set

    overlap = train_set.intersection(eval_set)
    if overlap:
        raise ValueError(
            f"Train/eval episode split cannot overlap. Overlapping ids: {sorted(overlap)}"
        )
    if not train_set:
        raise ValueError("Resolved train split is empty.")
    if not eval_set:
        raise ValueError("Resolved eval split is empty.")
    return sorted(train_set), sorted(eval_set)


def _oversample_hard_episodes(
    *,
    dataset: RLEnvDataset,
    hard_episode_ids: list[int],
    oversample_factor: int,
) -> RLEnvDataset:
    factor = max(1, int(oversample_factor))
    hard_ids = [int(value) for value in hard_episode_ids]
    if factor <= 1 or not hard_ids:
        return dataset

    tasks = dataset.tasks.copy().reset_index(drop=True)
    episodes = dataset.episodes.copy().reset_index(drop=True)
    available = set(int(value) for value in episodes["episode_id"].tolist())
    selected = [episode_id for episode_id in hard_ids if episode_id in available]
    if not selected:
        return dataset

    max_episode_id = int(max(available)) if available else 0
    cloned_task_frames: list[pd.DataFrame] = []
    cloned_episode_frames: list[pd.DataFrame] = []

    for source_episode_id in selected:
        source_tasks = tasks[tasks["episode_id"] == int(source_episode_id)].copy()
        source_episode = episodes[episodes["episode_id"] == int(source_episode_id)].copy()
        for clone_idx in range(factor - 1):
            max_episode_id += 1
            cloned_tasks = source_tasks.copy()
            cloned_tasks["episode_id"] = int(max_episode_id)
            cloned_tasks["task_id"] = (
                cloned_tasks["task_id"].astype(str)
                + f"__hardos_{source_episode_id}_{clone_idx + 1}"
            )
            cloned_episode = source_episode.copy()
            cloned_episode["episode_id"] = int(max_episode_id)
            if "source_kind" in cloned_episode.columns:
                cloned_episode["source_kind"] = (
                    cloned_episode["source_kind"].astype(str) + "::hard_oversample"
                )
            cloned_task_frames.append(cloned_tasks)
            cloned_episode_frames.append(cloned_episode)

    if not cloned_task_frames:
        return dataset

    merged_tasks = pd.concat([tasks, *cloned_task_frames], ignore_index=True)
    merged_episodes = pd.concat([episodes, *cloned_episode_frames], ignore_index=True)
    metadata = dict(dataset.metadata)
    metadata["hard_episode_oversample_factor"] = int(factor)
    metadata["hard_episode_oversampled_ids"] = selected
    return RLEnvDataset(
        tasks=merged_tasks,
        machines=dataset.machines.copy(),
        episodes=merged_episodes,
        metadata=metadata,
    )


def _build_learning_rate_schedule(
    *,
    schedule_name: str,
    base_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_timesteps: int,
) -> Any:
    normalized = str(schedule_name).strip().lower()
    base = float(base_learning_rate)
    lr_min = float(min_learning_rate)
    if normalized == "constant":
        def _constant_schedule(_: float) -> float:
            return base
        return _constant_schedule

    if normalized != "warmup_cosine":
        raise ValueError(
            f"Unsupported learning-rate schedule '{schedule_name}'. "
            "Expected one of: constant, warmup_cosine."
        )

    total_steps = max(1, int(total_timesteps))
    warmup = max(0, min(int(warmup_steps), total_steps - 1))

    def _schedule(progress_remaining: float) -> float:
        progress = float(np.clip(progress_remaining, 0.0, 1.0))
        current_step = (1.0 - progress) * float(total_steps)
        if warmup > 0 and current_step < float(warmup):
            alpha = current_step / float(warmup)
            return float(lr_min + (base - lr_min) * alpha)

        decay_denom = max(1.0, float(total_steps - warmup))
        decay_step = max(0.0, current_step - float(warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_step / decay_denom)))
        return float(lr_min + (base - lr_min) * cosine)

    return _schedule


def _resolve_reward_weights(args: argparse.Namespace) -> RewardWeights:
    base = RewardWeights()
    return RewardWeights(
        feasible_bonus=base.feasible_bonus,
        overload_penalty=base.overload_penalty,
        invalid_action_penalty=base.invalid_action_penalty,
        defer_penalty=base.defer_penalty
        if args.reward_defer_penalty is None
        else float(args.reward_defer_penalty),
        defer_escalation_rate=base.defer_escalation_rate
        if args.reward_defer_escalation_rate is None
        else float(args.reward_defer_escalation_rate),
        wait_penalty_weight=base.wait_penalty_weight
        if args.reward_wait_penalty_weight is None
        else float(args.reward_wait_penalty_weight),
        missed_deadline_penalty=base.missed_deadline_penalty,
        lateness_penalty_weight=base.lateness_penalty_weight,
        balance_bonus_weight=base.balance_bonus_weight,
        fragmentation_penalty_weight=base.fragmentation_penalty_weight,
        hotspot_penalty_weight=base.hotspot_penalty_weight,
        historical_match_bonus=base.historical_match_bonus,
        completion_bonus=base.completion_bonus,
        deadline_met_bonus=base.deadline_met_bonus,
        on_time_completion_bonus=base.on_time_completion_bonus,
        turnaround_penalty_weight=base.turnaround_penalty_weight
        if args.reward_turnaround_penalty_weight is None
        else float(args.reward_turnaround_penalty_weight),
        utilization_bonus_weight=base.utilization_bonus_weight
        if args.reward_utilization_bonus_weight is None
        else float(args.reward_utilization_bonus_weight),
        idle_machine_penalty_weight=base.idle_machine_penalty_weight
        if args.reward_idle_machine_penalty_weight is None
        else float(args.reward_idle_machine_penalty_weight),
        tail_wait_cvar_weight=base.tail_wait_cvar_weight
        if args.reward_tail_wait_cvar_weight is None
        else float(args.reward_tail_wait_cvar_weight),
        defer_guard_wait_ratio_threshold=base.defer_guard_wait_ratio_threshold
        if args.reward_defer_guard_wait_ratio_threshold is None
        else float(args.reward_defer_guard_wait_ratio_threshold),
        defer_guard_queue_pressure_threshold=base.defer_guard_queue_pressure_threshold
        if args.reward_defer_guard_queue_pressure_threshold is None
        else float(args.reward_defer_guard_queue_pressure_threshold),
        defer_guard_near_deadline_threshold=base.defer_guard_near_deadline_threshold
        if args.reward_defer_guard_near_deadline_threshold is None
        else float(args.reward_defer_guard_near_deadline_threshold),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MaskablePPO policy for the scheduler environment.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--init-model-path", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--progress-bar", action="store_true")

    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--train-episode-ids", type=str, default=None)
    parser.add_argument("--eval-episode-ids", type=str, default=None)
    parser.add_argument("--hard-episode-ids", type=str, default=None)
    parser.add_argument("--hard-episode-oversample-factor", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--metrics-window", type=int, default=1024)

    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--machine-capacity-scale", type=float, default=1.0)
    parser.add_argument("--machine-pool-size", type=int, default=None)
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-schedule", type=str, choices=["constant", "warmup_cosine"], default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=50_000)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden-dims", type=str, default="256,256")
    parser.add_argument("--features-extractor", type=str, choices=["default", "attention"], default="default")
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--attention-task-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-candidate-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-fleet-hidden-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.1)

    parser.add_argument("--reward-defer-penalty", type=float, default=None)
    parser.add_argument("--reward-defer-escalation-rate", type=float, default=None)
    parser.add_argument("--reward-wait-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-turnaround-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-utilization-bonus-weight", type=float, default=None)
    parser.add_argument("--reward-idle-machine-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-tail-wait-cvar-weight", type=float, default=None)
    parser.add_argument("--reward-defer-guard-wait-ratio-threshold", type=float, default=None)
    parser.add_argument("--reward-defer-guard-queue-pressure-threshold", type=float, default=None)
    parser.add_argument("--reward-defer-guard-near-deadline-threshold", type=float, default=None)

    return parser


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    return serializable


def main() -> None:
    args = _build_parser().parse_args()
    dependencies = _require_rl_dependencies()
    MaskablePPO = dependencies["MaskablePPO"]
    MaskableEvalCallback = dependencies["MaskableEvalCallback"]
    CallbackList = dependencies["CallbackList"]
    CheckpointCallback = dependencies["CheckpointCallback"]
    DummyVecEnv = dependencies["DummyVecEnv"]
    get_schedule_fn = dependencies["get_schedule_fn"]
    Monitor = dependencies["Monitor"]
    VecMonitor = dependencies["VecMonitor"]

    hidden_dims = _parse_hidden_dims(args.policy_hidden_dims)
    reward_weights = _resolve_reward_weights(args)
    lr_schedule = _build_learning_rate_schedule(
        schedule_name=str(args.learning_rate_schedule),
        base_learning_rate=float(args.learning_rate),
        min_learning_rate=float(args.lr_min),
        warmup_steps=int(args.lr_warmup_steps),
        total_timesteps=int(args.total_timesteps),
    )

    n_envs = max(1, int(args.n_envs))
    rollout_batch_size = int(args.n_steps) * n_envs
    if int(args.batch_size) > rollout_batch_size:
        raise ValueError(
            "batch-size cannot exceed n-steps * n-envs for PPO rollout collection. "
            f"Received batch-size={args.batch_size}, n-steps={args.n_steps}, n-envs={n_envs}."
        )

    run_dir = _build_run_dir(args.output_dir, args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = run_dir / "eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_scheduler_dataset(args.dataset)
    train_episode_ids, eval_episode_ids = _resolve_episode_split(
        dataset=dataset,
        seed=int(args.seed),
        eval_fraction=float(args.eval_fraction),
        train_episode_ids_arg=args.train_episode_ids,
        eval_episode_ids_arg=args.eval_episode_ids,
    )
    train_dataset = subset_dataset_by_episode_ids(dataset, train_episode_ids)
    eval_dataset = subset_dataset_by_episode_ids(dataset, eval_episode_ids)
    hard_episode_ids = _parse_episode_ids(args.hard_episode_ids)
    hard_train_ids = [episode_id for episode_id in hard_episode_ids if episode_id in set(train_episode_ids)]
    train_dataset = _oversample_hard_episodes(
        dataset=train_dataset,
        hard_episode_ids=hard_train_ids,
        oversample_factor=int(args.hard_episode_oversample_factor),
    )

    def wrap_for_training(env: Any) -> Any:
        return ActionMaskInfoWrapper(Monitor(env))

    train_env_fns = []
    for env_idx in range(n_envs):
        env_seed = int(args.seed) + env_idx
        train_env_fns.append(
            lambda env_seed=env_seed: wrap_for_training(
                build_scheduler_env(
                    dataset=train_dataset,
                    top_k_candidates=int(args.top_k_candidates),
                    max_steps=int(args.max_steps),
                    max_consecutive_defers=int(args.max_consecutive_defers),
                    invalid_action_limit=int(args.invalid_action_limit),
                    machine_capacity_scale=float(args.machine_capacity_scale),
                    machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
                    deadline_slack_factor=float(args.deadline_slack_factor),
                    reward_weights=reward_weights,
                    random_state=env_seed,
                    randomize_on_reset=True,
                )
            )
        )

    train_env = VecMonitor(DummyVecEnv(train_env_fns))
    eval_env = wrap_for_training(
        build_scheduler_env(
            dataset=eval_dataset,
            top_k_candidates=int(args.top_k_candidates),
            max_steps=int(args.max_steps),
            max_consecutive_defers=int(args.max_consecutive_defers),
            invalid_action_limit=int(args.invalid_action_limit),
            machine_capacity_scale=float(args.machine_capacity_scale),
            machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
            deadline_slack_factor=float(args.deadline_slack_factor),
            reward_weights=reward_weights,
            random_state=int(args.seed) + 10_000,
            randomize_on_reset=False,
        )
    )

    policy_kwargs = {
        "net_arch": {
            "pi": hidden_dims,
            "vf": hidden_dims,
        }
    }
    if str(args.features_extractor) == "attention":
        try:
            from .attention_extractor import SchedulerAttentionExtractor
        except ImportError:
            from attention_extractor import SchedulerAttentionExtractor
        policy_kwargs["features_extractor_class"] = SchedulerAttentionExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": int(args.features_dim),
            "task_hidden_dim": int(args.attention_task_hidden_dim),
            "candidate_hidden_dim": int(args.attention_candidate_hidden_dim),
            "fleet_hidden_dim": int(args.attention_fleet_hidden_dim),
            "attention_heads": int(args.attention_heads),
            "attention_dropout": float(args.attention_dropout),
        }

    tensorboard_log: str | None = str(run_dir / "tensorboard")
    try:
        import tensorboard  # noqa: F401
    except ModuleNotFoundError:
        tensorboard_log = None
        print(
            "TensorBoard is not installed; proceeding without tensorboard logging. "
            "Install with: pip install tensorboard"
        )

    if args.init_model_path is not None:
        model = MaskablePPO.load(
            str(args.init_model_path),
            env=train_env,
            device=str(args.device),
        )
        if int(args.n_steps) != int(model.n_steps):
            raise ValueError(
                "n-steps must match the loaded model's rollout buffer size when fine-tuning. "
                f"Loaded model n-steps={model.n_steps}, requested n-steps={args.n_steps}."
            )
        model.learning_rate = lr_schedule
        model.lr_schedule = lr_schedule
        model.batch_size = int(args.batch_size)
        model.n_epochs = int(args.n_epochs)
        model.gamma = float(args.gamma)
        model.gae_lambda = float(args.gae_lambda)
        model.clip_range = get_schedule_fn(float(args.clip_range))
        model.ent_coef = float(args.ent_coef)
        model.vf_coef = float(args.vf_coef)
        model.max_grad_norm = float(args.max_grad_norm)
        model.tensorboard_log = tensorboard_log
        model.verbose = 1
        if str(args.features_extractor) == "attention":
            loaded_extractor_name = type(model.policy.features_extractor).__name__
            if loaded_extractor_name != "SchedulerAttentionExtractor":
                raise ValueError(
                    "Loaded checkpoint is not using SchedulerAttentionExtractor, "
                    "so attention fine-tuning cannot be enabled on this run. "
                    f"Loaded extractor: {loaded_extractor_name}"
                )
        print(f"Loaded initial model weights from: {args.init_model_path}")
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=lr_schedule,
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            seed=int(args.seed),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            device=str(args.device),
            verbose=1,
        )

    callbacks = [
        SchedulerMetricsCallback(window_size=int(args.metrics_window)),
    ]

    if int(args.checkpoint_freq) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_freq) // n_envs),
                save_path=str(checkpoint_dir),
                name_prefix="ppo_scheduler",
            )
        )

    eval_callback = None
    if int(args.eval_freq) > 0 and int(args.eval_episodes) > 0:
        eval_callback = MaskableEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(eval_log_dir),
            eval_freq=max(1, int(args.eval_freq) // n_envs),
            n_eval_episodes=int(args.eval_episodes),
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    callback = CallbackList(callbacks)

    learn_kwargs: dict[str, Any] = {
        "total_timesteps": int(args.total_timesteps),
        "callback": callback,
        "tb_log_name": run_dir.name,
    }
    if args.progress_bar:
        learn_kwargs["progress_bar"] = True

    try:
        model.learn(**learn_kwargs)
    except TypeError as exc:
        # Backward compatibility for stable-baselines3 versions without `progress_bar`.
        if "progress_bar" in learn_kwargs and "progress_bar" in str(exc):
            learn_kwargs.pop("progress_bar", None)
            model.learn(**learn_kwargs)
        else:
            raise

    model_path = run_dir / "final_model"
    model.save(str(model_path))

    config_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": _serialize_args(args),
        "resolved": {
            "dataset_path": str(Path(args.dataset)),
            "train_episode_ids": train_episode_ids,
            "eval_episode_ids": eval_episode_ids,
            "hard_episode_ids": hard_episode_ids,
            "hard_episode_oversample_factor": int(args.hard_episode_oversample_factor),
            "policy_hidden_dims": hidden_dims,
            "n_envs": n_envs,
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    summary_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.with_suffix(".zip")),
        "total_timesteps": int(args.total_timesteps),
        "train_episode_count": len(train_episode_ids),
        "eval_episode_count": len(eval_episode_ids),
        "best_mean_reward": None if eval_callback is None else float(eval_callback.best_mean_reward),
        "run_dir": str(run_dir),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    train_env.close()
    eval_env.close()

    print(f"Training complete. Run directory: {run_dir}")
    print(f"Final model: {model_path.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
