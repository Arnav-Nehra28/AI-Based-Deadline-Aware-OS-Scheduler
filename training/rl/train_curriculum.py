from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from .env_factory import DEFAULT_RL_DATASET_PATH
except ImportError:
    from env_factory import DEFAULT_RL_DATASET_PATH


def _parse_csv(raw: str, cast: Any) -> list[Any]:
    values: list[Any] = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(cast(token))
    if not values:
        raise ValueError("Curriculum list arguments cannot be empty.")
    return values


def _resolve_phase_dir(output_dir: Path, expected_name: str) -> Path:
    exact = output_dir / expected_name
    if exact.exists():
        return exact
    matches = sorted(output_dir.glob(f"{expected_name}*"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"Unable to locate phase run directory matching '{expected_name}'.")
    return matches[-1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged curriculum training for the RL scheduler policy.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)
    parser.add_argument("--policy-hidden-dims", type=str, default="512,256,128")
    parser.add_argument("--features-extractor", type=str, choices=["default", "attention"], default="attention")
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--attention-task-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-candidate-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-fleet-hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate-schedule", type=str, choices=["constant", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--metrics-window", type=int, default=1024)
    parser.add_argument("--progress-bar", action="store_true")

    parser.add_argument("--phase-names", type=str, default="warmup,mixed,stress")
    parser.add_argument("--phase-timesteps", type=str, default="100000,100000,200000")
    parser.add_argument("--phase-machine-capacity-scales", type=str, default="0.8,0.5,0.35")
    parser.add_argument("--phase-machine-pool-sizes", type=str, default="128,96,32")
    parser.add_argument("--phase-learning-rates", type=str, default="3e-4,1e-4,5e-5")
    parser.add_argument("--phase-ent-coefs", type=str, default="0.03,0.02,0.01")
    parser.add_argument("--phase-lr-warmup-steps", type=str, default="25000,25000,50000")

    parser.add_argument("--reward-defer-penalty", type=float, default=None)
    parser.add_argument("--reward-defer-escalation-rate", type=float, default=None)
    parser.add_argument("--reward-wait-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-turnaround-penalty-weight", type=float, default=None)
    parser.add_argument("--reward-utilization-bonus-weight", type=float, default=None)
    parser.add_argument("--reward-idle-machine-penalty-weight", type=float, default=None)
    return parser


def _optional_arg(flag: str, value: Any) -> list[str]:
    if value is None:
        return []
    return [flag, str(value)]


def main() -> None:
    args = _build_parser().parse_args()
    phase_names = _parse_csv(args.phase_names, str)
    phase_timesteps = _parse_csv(args.phase_timesteps, int)
    phase_scales = _parse_csv(args.phase_machine_capacity_scales, float)
    phase_pools = _parse_csv(args.phase_machine_pool_sizes, int)
    phase_lrs = _parse_csv(args.phase_learning_rates, float)
    phase_ent = _parse_csv(args.phase_ent_coefs, float)
    phase_warmups = _parse_csv(args.phase_lr_warmup_steps, int)

    phase_count = len(phase_names)
    lengths = [len(phase_timesteps), len(phase_scales), len(phase_pools), len(phase_lrs), len(phase_ent), len(phase_warmups)]
    if any(length != phase_count for length in lengths):
        raise ValueError(
            "All phase lists must have equal length. "
            f"names={phase_count}, timesteps={len(phase_timesteps)}, scales={len(phase_scales)}, "
            f"pools={len(phase_pools)}, lrs={len(phase_lrs)}, ent={len(phase_ent)}, warmups={len(phase_warmups)}."
        )

    run_prefix = args.run_name or f"curriculum_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_model_path: Path | None = None
    phase_records: list[dict[str, Any]] = []
    for index in range(phase_count):
        phase_name = str(phase_names[index])
        phase_run_name = f"{run_prefix}_{index + 1:02d}_{phase_name}"
        phase_cmd = [
            sys.executable,
            "-m",
            "training.rl.train_ppo",
            "--dataset",
            str(args.dataset),
            "--output-dir",
            str(output_dir),
            "--run-name",
            phase_run_name,
            "--total-timesteps",
            str(int(phase_timesteps[index])),
            "--seed",
            str(int(args.seed) + index),
            "--device",
            str(args.device),
            "--n-envs",
            str(int(args.n_envs)),
            "--eval-fraction",
            str(float(args.eval_fraction)),
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
            str(float(phase_scales[index])),
            "--machine-pool-size",
            str(int(phase_pools[index])),
            "--deadline-slack-factor",
            str(float(args.deadline_slack_factor)),
            "--learning-rate",
            str(float(phase_lrs[index])),
            "--learning-rate-schedule",
            str(args.learning_rate_schedule),
            "--lr-warmup-steps",
            str(int(phase_warmups[index])),
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
            str(float(phase_ent[index])),
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
            "--attention-heads",
            str(int(args.attention_heads)),
            "--attention-dropout",
            str(float(args.attention_dropout)),
            "--attention-task-hidden-dim",
            str(int(args.attention_task_hidden_dim)),
            "--attention-candidate-hidden-dim",
            str(int(args.attention_candidate_hidden_dim)),
            "--attention-fleet-hidden-dim",
            str(int(args.attention_fleet_hidden_dim)),
        ]
        phase_cmd.extend(_optional_arg("--init-model-path", init_model_path))
        phase_cmd.extend(_optional_arg("--reward-defer-penalty", args.reward_defer_penalty))
        phase_cmd.extend(_optional_arg("--reward-defer-escalation-rate", args.reward_defer_escalation_rate))
        phase_cmd.extend(_optional_arg("--reward-wait-penalty-weight", args.reward_wait_penalty_weight))
        phase_cmd.extend(_optional_arg("--reward-turnaround-penalty-weight", args.reward_turnaround_penalty_weight))
        phase_cmd.extend(_optional_arg("--reward-utilization-bonus-weight", args.reward_utilization_bonus_weight))
        phase_cmd.extend(_optional_arg("--reward-idle-machine-penalty-weight", args.reward_idle_machine_penalty_weight))
        if bool(args.progress_bar):
            phase_cmd.append("--progress-bar")

        subprocess.run(phase_cmd, check=True)
        phase_dir = _resolve_phase_dir(output_dir=output_dir, expected_name=phase_run_name)
        init_model_path = phase_dir / "final_model.zip"
        phase_records.append(
            {
                "phase_index": int(index + 1),
                "phase_name": phase_name,
                "run_dir": str(phase_dir),
                "model_path": str(init_model_path),
                "timesteps": int(phase_timesteps[index]),
                "machine_capacity_scale": float(phase_scales[index]),
                "machine_pool_size": int(phase_pools[index]),
                "learning_rate": float(phase_lrs[index]),
                "ent_coef": float(phase_ent[index]),
                "lr_warmup_steps": int(phase_warmups[index]),
            }
        )

    final_model_path = init_model_path
    if final_model_path is None:
        raise RuntimeError("Curriculum run finished without producing a final model.")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_prefix": run_prefix,
        "dataset_path": str(args.dataset),
        "final_model_path": str(final_model_path),
        "phases": phase_records,
    }
    summary_path = output_dir / f"{run_prefix}_curriculum_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Curriculum training complete. Final model: {final_model_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
