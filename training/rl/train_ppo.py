from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    train_episode_ids, eval_episode_ids = split_episode_ids(
        dataset,
        eval_fraction=float(args.eval_fraction),
        seed=int(args.seed),
    )
    train_dataset = subset_dataset_by_episode_ids(dataset, train_episode_ids)
    eval_dataset = subset_dataset_by_episode_ids(dataset, eval_episode_ids)

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
        model.learning_rate = float(args.learning_rate)
        model.lr_schedule = get_schedule_fn(float(args.learning_rate))
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
        print(f"Loaded initial model weights from: {args.init_model_path}")
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=float(args.learning_rate),
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
