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
except ImportError:
    from callbacks import SchedulerMetricsCallback
    from env_factory import (
        DEFAULT_RL_DATASET_PATH,
        build_scheduler_env,
        load_scheduler_dataset,
        split_episode_ids,
        subset_dataset_by_episode_ids,
    )

from rl_pipeline.environment import RewardWeights


def _require_dependencies() -> dict[str, Any]:
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for RecurrentPPO training. Install with:\n"
            "pip install gymnasium stable-baselines3 sb3-contrib"
        ) from exc
    return {
        "RecurrentPPO": RecurrentPPO,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "VecMonitor": VecMonitor,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a recurrent PPO (LSTM) scheduler policy.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--metrics-window", type=int, default=1024)

    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--machine-capacity-scale", type=float, default=0.35)
    parser.add_argument("--machine-pool-size", type=int, default=32)
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--n-lstm-layers", type=int, default=1)

    parser.add_argument("--reward-defer-penalty", type=float, default=None)
    parser.add_argument("--reward-defer-escalation-rate", type=float, default=None)
    parser.add_argument("--reward-wait-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-turnaround-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-utilization-bonus-weight", type=float, default=None)
    parser.add_argument("--reward-idle-machine-penalty-weight", type=float, default=None)
    return parser


def _build_run_dir(output_root: Path, run_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = run_name or f"recurrent_ppo_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolve_reward_weights(args: argparse.Namespace) -> RewardWeights:
    base = RewardWeights()
    return RewardWeights(
        feasible_bonus=base.feasible_bonus,
        overload_penalty=base.overload_penalty,
        invalid_action_penalty=base.invalid_action_penalty,
        defer_penalty=base.defer_penalty if args.reward_defer_penalty is None else float(args.reward_defer_penalty),
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
    )


def main() -> None:
    args = _build_parser().parse_args()
    dependencies = _require_dependencies()
    RecurrentPPO = dependencies["RecurrentPPO"]
    CallbackList = dependencies["CallbackList"]
    CheckpointCallback = dependencies["CheckpointCallback"]
    Monitor = dependencies["Monitor"]
    DummyVecEnv = dependencies["DummyVecEnv"]
    VecMonitor = dependencies["VecMonitor"]

    n_envs = max(1, int(args.n_envs))
    run_dir = _build_run_dir(Path(args.output_dir), args.run_name)
    reward_weights = _resolve_reward_weights(args)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_scheduler_dataset(args.dataset)
    train_episode_ids, eval_episode_ids = split_episode_ids(
        dataset,
        eval_fraction=float(args.eval_fraction),
        seed=int(args.seed),
    )
    train_dataset = subset_dataset_by_episode_ids(dataset, train_episode_ids)
    _ = subset_dataset_by_episode_ids(dataset, eval_episode_ids)

    train_env_fns = []
    for env_idx in range(n_envs):
        env_seed = int(args.seed) + env_idx
        train_env_fns.append(
            lambda env_seed=env_seed: Monitor(
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

    policy_kwargs = {
        "lstm_hidden_size": int(args.lstm_hidden_size),
        "n_lstm_layers": int(args.n_lstm_layers),
    }
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
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
        policy_kwargs=policy_kwargs,
        seed=int(args.seed),
        tensorboard_log=str(run_dir / "tensorboard"),
        device=str(args.device),
        verbose=1,
    )

    callbacks = [SchedulerMetricsCallback(window_size=int(args.metrics_window))]
    if int(args.checkpoint_freq) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_freq) // n_envs),
                save_path=str(checkpoint_dir),
                name_prefix="recurrent_ppo_scheduler",
            )
        )
    callback = CallbackList(callbacks)

    learn_kwargs: dict[str, Any] = {
        "total_timesteps": int(args.total_timesteps),
        "callback": callback,
        "tb_log_name": run_dir.name,
    }
    if bool(args.progress_bar):
        learn_kwargs["progress_bar"] = True
    try:
        model.learn(**learn_kwargs)
    except TypeError as exc:
        if "progress_bar" in learn_kwargs and "progress_bar" in str(exc):
            learn_kwargs.pop("progress_bar", None)
            model.learn(**learn_kwargs)
        else:
            raise

    model_path = run_dir / "final_model"
    model.save(str(model_path))
    train_env.close()

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.with_suffix(".zip")),
        "run_dir": str(run_dir),
        "train_episode_count": len(train_episode_ids),
        "eval_episode_count": len(eval_episode_ids),
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "note": "RecurrentPPO does not use action masking; invalid actions are handled via environment penalties.",
    }
    (run_dir / "train_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Recurrent training complete. Run directory: {run_dir}")
    print(f"Final model: {model_path.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
