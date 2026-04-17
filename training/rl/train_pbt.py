from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

try:
    from .env_factory import DEFAULT_RL_DATASET_PATH
except ImportError:
    from env_factory import DEFAULT_RL_DATASET_PATH


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _mutate(base: dict[str, float], rng: random.Random) -> dict[str, float]:
    return {
        "learning_rate": _clamp(base["learning_rate"] * rng.uniform(0.7, 1.3), 1e-5, 5e-4),
        "ent_coef": _clamp(base["ent_coef"] * rng.uniform(0.6, 1.4), 0.001, 0.08),
        "reward_defer_penalty": _clamp(base["reward_defer_penalty"] * rng.uniform(0.8, 1.25), -4.5, -1.0),
        "reward_wait_penalty_weight": _clamp(base["reward_wait_penalty_weight"] * rng.uniform(0.8, 1.3), 0.8, 4.0),
        "reward_turnaround_penalty_weight": _clamp(
            base["reward_turnaround_penalty_weight"] * rng.uniform(0.8, 1.3),
            0.5,
            3.0,
        ),
        "reward_utilization_bonus_weight": _clamp(
            base["reward_utilization_bonus_weight"] * rng.uniform(0.7, 1.3),
            0.1,
            1.5,
        ),
        "reward_idle_machine_penalty_weight": _clamp(
            base["reward_idle_machine_penalty_weight"] * rng.uniform(0.8, 1.25),
            0.1,
            1.0,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Population-based training orchestrator for scheduler PPO.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rl"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timesteps-per-round", type=int, default=120_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--machine-capacity-scale", type=float, default=0.35)
    parser.add_argument("--machine-pool-size", type=int, default=32)
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)
    parser.add_argument("--policy-hidden-dims", type=str, default="512,256,128")
    parser.add_argument("--features-extractor", type=str, choices=["default", "attention"], default="attention")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate-schedule", type=str, choices=["constant", "warmup_cosine"], default="warmup_cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=30_000)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--metrics-window", type=int, default=1024)
    return parser


def _read_best_mean_reward(run_dir: Path) -> float:
    summary_path = run_dir / "train_summary.json"
    if not summary_path.exists():
        return float("-inf")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    value = payload.get("best_mean_reward")
    if value is None:
        return float("-inf")
    return float(value)


def _resolve_run_dir(output_dir: Path, expected_name: str) -> Path:
    direct = output_dir / expected_name
    if direct.exists():
        return direct
    matches = sorted(output_dir.glob(f"{expected_name}*"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"Could not resolve run directory for '{expected_name}'.")
    return matches[-1]


def main() -> None:
    args = _build_parser().parse_args()
    rng = random.Random(int(args.seed))
    population_size = max(2, int(args.population_size))
    rounds = max(1, int(args.rounds))
    run_prefix = args.run_name or f"pbt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_hparams = {
        "learning_rate": 1e-4,
        "ent_coef": 0.015,
        "reward_defer_penalty": -2.5,
        "reward_wait_penalty_weight": 2.5,
        "reward_turnaround_penalty_weight": 1.5,
        "reward_utilization_bonus_weight": 0.8,
        "reward_idle_machine_penalty_weight": 0.5,
    }
    members = [dict(base_hparams) for _ in range(population_size)]
    member_models: list[Path | None] = [None for _ in range(population_size)]
    history: list[dict[str, Any]] = []

    for round_idx in range(rounds):
        round_records: list[dict[str, Any]] = []
        if round_idx > 0:
            scored_prev = sorted(
                history[-1]["members"], key=lambda row: float(row["best_mean_reward"]), reverse=True
            )
            elite = scored_prev[0]
            elite_params = dict(elite["hparams"])
            elite_model = Path(elite["model_path"])
            for member_idx in range(population_size):
                if member_idx == 0:
                    members[member_idx] = elite_params
                    member_models[member_idx] = elite_model
                else:
                    members[member_idx] = _mutate(elite_params, rng)
                    member_models[member_idx] = elite_model

        for member_idx in range(population_size):
            run_name = f"{run_prefix}_r{round_idx + 1:02d}_m{member_idx + 1:02d}"
            hparams = members[member_idx]
            cmd = [
                sys.executable,
                "-m",
                "training.rl.train_ppo",
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(output_dir),
                "--run-name",
                run_name,
                "--total-timesteps",
                str(int(args.timesteps_per_round)),
                "--seed",
                str(int(args.seed) + (round_idx * 100) + member_idx),
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
                str(float(args.machine_capacity_scale)),
                "--machine-pool-size",
                str(int(args.machine_pool_size)),
                "--deadline-slack-factor",
                str(float(args.deadline_slack_factor)),
                "--learning-rate",
                str(float(hparams["learning_rate"])),
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
                str(float(hparams["ent_coef"])),
                "--vf-coef",
                str(float(args.vf_coef)),
                "--max-grad-norm",
                str(float(args.max_grad_norm)),
                "--policy-hidden-dims",
                str(args.policy_hidden_dims),
                "--features-extractor",
                str(args.features_extractor),
                "--reward-defer-penalty",
                str(float(hparams["reward_defer_penalty"])),
                "--reward-wait-penalty-weight",
                str(float(hparams["reward_wait_penalty_weight"])),
                "--reward-turnaround-penalty-weight",
                str(float(hparams["reward_turnaround_penalty_weight"])),
                "--reward-utilization-bonus-weight",
                str(float(hparams["reward_utilization_bonus_weight"])),
                "--reward-idle-machine-penalty-weight",
                str(float(hparams["reward_idle_machine_penalty_weight"])),
            ]
            if member_models[member_idx] is not None:
                cmd.extend(["--init-model-path", str(member_models[member_idx])])
            subprocess.run(cmd, check=True)
            run_dir = _resolve_run_dir(output_dir=output_dir, expected_name=run_name)
            model_path = run_dir / "final_model.zip"
            best_mean_reward = _read_best_mean_reward(run_dir)
            member_models[member_idx] = model_path
            round_records.append(
                {
                    "member_index": int(member_idx),
                    "run_dir": str(run_dir),
                    "model_path": str(model_path),
                    "best_mean_reward": float(best_mean_reward),
                    "hparams": dict(hparams),
                }
            )

        history.append(
            {
                "round_index": int(round_idx + 1),
                "members": round_records,
            }
        )

    final_members = sorted(history[-1]["members"], key=lambda row: float(row["best_mean_reward"]), reverse=True)
    champion = final_members[0]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_prefix": run_prefix,
        "rounds": int(rounds),
        "population_size": int(population_size),
        "history": history,
        "champion": champion,
    }
    summary_path = output_dir / f"{run_prefix}_pbt_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PBT complete. Champion model: {champion['model_path']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
