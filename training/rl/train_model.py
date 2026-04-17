from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from .env_factory import DEFAULT_RL_DATASET_PATH, load_scheduler_dataset, subset_dataset_by_episode_ids
    from .experiment_splits import save_episode_splits, split_episode_ids_train_val_test
except ImportError:
    from env_factory import DEFAULT_RL_DATASET_PATH, load_scheduler_dataset, subset_dataset_by_episode_ids
    from experiment_splits import save_episode_splits, split_episode_ids_train_val_test

from rl_pipeline.env_dataset import save_env_dataset


def _build_run_name() -> str:
    return f"ppo_exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the scheduler RL model with deterministic train/val/test episode splits "
            "and persist split metadata for downstream validate/test/evaluate commands."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--init-model-path", type=Path, default=None)

    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--progress-bar", action="store_true")

    parser.add_argument("--n-envs", type=int, default=4)
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
    return parser


def _resolve_run_dir(
    *,
    output_dir: Path,
    prepared_dir_name: str,
    before_dirs: set[Path],
    run_name: str,
) -> Path:
    after_dirs = {
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name != prepared_dir_name
    }
    created_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime)
    if created_dirs:
        return created_dirs[-1]

    matching_dirs = sorted(
        [path for path in after_dirs if path.name.startswith(run_name)],
        key=lambda path: path.stat().st_mtime,
    )
    if not matching_dirs:
        raise RuntimeError(
            "Could not resolve the PPO run directory after training. "
            f"Looked under: {output_dir}"
        )
    return matching_dirs[-1]


def _as_serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def main() -> None:
    args = _build_parser().parse_args()
    run_name = args.run_name or _build_run_name()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_dir = output_dir / "_prepared_datasets"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    before_dirs = {
        path for path in output_dir.iterdir() if path.is_dir() and path.name != prepared_dir.name
    }

    dataset = load_scheduler_dataset(args.dataset)
    splits = split_episode_ids_train_val_test(
        dataset,
        train_fraction=float(args.train_fraction),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        seed=int(args.split_seed),
    )

    train_val_ids = sorted(
        [*splits.train_episode_ids, *splits.val_episode_ids]
    )
    train_val_dataset = subset_dataset_by_episode_ids(dataset, train_val_ids)
    prepared_dataset_path = prepared_dir / f"{run_name}_train_val_dataset.json.gz"
    save_env_dataset(train_val_dataset, prepared_dataset_path)

    eval_fraction = float(len(splits.val_episode_ids) / max(len(train_val_ids), 1))

    command = [
        sys.executable,
        "-m",
        "training.rl.train_ppo",
        "--dataset",
        str(prepared_dataset_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--total-timesteps",
        str(int(args.total_timesteps)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--n-envs",
        str(int(args.n_envs)),
        "--eval-fraction",
        f"{eval_fraction:.12f}",
        "--eval-episodes",
        str(int(args.eval_episodes)),
        "--eval-freq",
        str(int(args.eval_freq)),
        "--checkpoint-freq",
        str(int(args.checkpoint_freq)),
        "--metrics-window",
        str(int(args.metrics_window)),
        "--top-k-candidates",
        str(int(args.top_k_candidates)),
        "--max-steps",
        str(int(args.max_steps)),
        "--max-consecutive-defers",
        str(int(args.max_consecutive_defers)),
        "--invalid-action-limit",
        str(int(args.invalid_action_limit)),
        "--machine-capacity-scale",
        str(float(args.machine_capacity_scale)),
        "--deadline-slack-factor",
        str(float(args.deadline_slack_factor)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--learning-rate-schedule",
        str(args.learning_rate_schedule),
        "--lr-warmup-steps",
        str(int(args.lr_warmup_steps)),
        "--lr-min",
        str(float(args.lr_min)),
        "--gamma",
        str(float(args.gamma)),
        "--gae-lambda",
        str(float(args.gae_lambda)),
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--clip-range",
        str(float(args.clip_range)),
        "--ent-coef",
        str(float(args.ent_coef)),
        "--vf-coef",
        str(float(args.vf_coef)),
        "--max-grad-norm",
        str(float(args.max_grad_norm)),
        "--policy-hidden-dims",
        str(args.policy_hidden_dims),
        "--features-extractor",
        str(args.features_extractor),
        "--features-dim",
        str(int(args.features_dim)),
        "--attention-task-hidden-dim",
        str(int(args.attention_task_hidden_dim)),
        "--attention-candidate-hidden-dim",
        str(int(args.attention_candidate_hidden_dim)),
        "--attention-fleet-hidden-dim",
        str(int(args.attention_fleet_hidden_dim)),
        "--attention-heads",
        str(int(args.attention_heads)),
        "--attention-dropout",
        str(float(args.attention_dropout)),
    ]
    if args.init_model_path is not None:
        command.extend(["--init-model-path", str(Path(args.init_model_path))])
    if args.machine_pool_size is not None:
        command.extend(["--machine-pool-size", str(int(args.machine_pool_size))])
    if bool(args.progress_bar):
        command.append("--progress-bar")
    if args.reward_defer_penalty is not None:
        command.extend(["--reward-defer-penalty", str(float(args.reward_defer_penalty))])
    if args.reward_defer_escalation_rate is not None:
        command.extend(["--reward-defer-escalation-rate", str(float(args.reward_defer_escalation_rate))])
    if args.reward_wait_penalty_weight is not None:
        command.extend(["--reward-wait-penalty-weight", str(float(args.reward_wait_penalty_weight))])
    if args.reward_turnaround_penalty_weight is not None:
        command.extend(["--reward-turnaround-penalty-weight", str(float(args.reward_turnaround_penalty_weight))])
    if args.reward_utilization_bonus_weight is not None:
        command.extend(["--reward-utilization-bonus-weight", str(float(args.reward_utilization_bonus_weight))])
    if args.reward_idle_machine_penalty_weight is not None:
        command.extend(["--reward-idle-machine-penalty-weight", str(float(args.reward_idle_machine_penalty_weight))])

    subprocess.run(command, check=True)

    run_dir = _resolve_run_dir(
        output_dir=output_dir,
        prepared_dir_name=prepared_dir.name,
        before_dirs=before_dirs,
        run_name=run_name,
    )

    split_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(args.dataset)),
        "prepared_train_val_dataset_path": str(prepared_dataset_path),
        "run_dir": str(run_dir),
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
        "split_seed": int(args.split_seed),
        "eval_fraction_used_for_training": float(eval_fraction),
        "train_model_args": _as_serializable_args(args),
    }
    split_path = run_dir / "episode_splits.json"
    save_episode_splits(
        splits=splits,
        output_path=split_path,
        metadata=split_metadata,
    )

    run_summary = {
        "run_dir": str(run_dir),
        "model_path": str(run_dir / "final_model.zip"),
        "split_path": str(split_path),
        "train_episode_count": len(splits.train_episode_ids),
        "val_episode_count": len(splits.val_episode_ids),
        "test_episode_count": len(splits.test_episode_ids),
    }
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    print("Train/val/test model training complete.")
    print(f"Run directory: {run_dir}")
    print(f"Model path: {run_dir / 'final_model.zip'}")
    print(f"Episode splits: {split_path}")
    print("")
    print("Next commands:")
    print(f"python -m training.rl.validate_model --run-dir {run_dir}")
    print(f"python -m training.rl.test_model --run-dir {run_dir}")
    print(f"python -m training.rl.evaluate_model --run-dir {run_dir}")


if __name__ == "__main__":
    main()
