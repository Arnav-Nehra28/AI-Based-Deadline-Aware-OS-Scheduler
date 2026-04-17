from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    from .env_factory import DEFAULT_RL_DATASET_PATH, load_scheduler_dataset
    from .evaluation_core import (
        comparison_row_from_summary,
        evaluate_heuristic_policy_on_episode_ids,
        evaluate_rl_policy_on_episode_ids,
    )
except ImportError:
    from env_factory import DEFAULT_RL_DATASET_PATH, load_scheduler_dataset
    from evaluation_core import (
        comparison_row_from_summary,
        evaluate_heuristic_policy_on_episode_ids,
        evaluate_rl_policy_on_episode_ids,
    )


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Seed list cannot be empty.")
    return values


def _parse_episode_ids(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-seed RL scheduler evaluation and report mean±std.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--episode-ids", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="13,23,37,42,77")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)
    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--machine-capacity-scale", type=float, default=1.0)
    parser.add_argument("--machine-pool-size", type=int, default=None)
    parser.add_argument("--use-hybrid-rl", action="store_true")
    parser.add_argument("--hybrid-defer-wait-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--hybrid-high-utilization-threshold", type=float, default=0.90)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _agg(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_split_payload(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    split_path = args.split_path or (run_dir / "episode_splits.json")
    split_payload = _load_split_payload(split_path)

    dataset_path = args.dataset or Path(str(split_payload.get("dataset_path", DEFAULT_RL_DATASET_PATH)))
    model_path = args.model_path or (run_dir / "final_model.zip")
    requested_ids = _parse_episode_ids(args.episode_ids)
    episode_ids = requested_ids or [int(value) for value in split_payload.get("test_episode_ids", [])]
    if not episode_ids:
        raise ValueError("No episode ids provided and no test episode ids found.")
    seeds = _parse_csv_ints(args.seeds)

    dataset = load_scheduler_dataset(dataset_path)
    fcfs = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset,
        episode_ids=episode_ids,
        heuristic="FCFS",
        deadline_slack_factor=float(args.deadline_slack_factor),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )
    sjf = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset,
        episode_ids=episode_ids,
        heuristic="SJF",
        deadline_slack_factor=float(args.deadline_slack_factor),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )
    rr = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset,
        episode_ids=episode_ids,
        heuristic="RR",
        deadline_slack_factor=float(args.deadline_slack_factor),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        rl = evaluate_rl_policy_on_episode_ids(
            model_path=model_path,
            dataset=dataset,
            episode_ids=episode_ids,
            seed=int(seed),
            device=str(args.device),
            stochastic=bool(args.stochastic),
            deadline_slack_factor=float(args.deadline_slack_factor),
            top_k_candidates=int(args.top_k_candidates),
            max_steps=int(args.max_steps),
            max_consecutive_defers=int(args.max_consecutive_defers),
            invalid_action_limit=int(args.invalid_action_limit),
            machine_capacity_scale=float(args.machine_capacity_scale),
            machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
            use_hybrid_scheduler=bool(args.use_hybrid_rl),
            hybrid_defer_wait_ratio_threshold=float(args.hybrid_defer_wait_ratio_threshold),
            hybrid_high_utilization_threshold=float(args.hybrid_high_utilization_threshold),
        )
        summary = rl["summary"]
        per_seed.append(
            {
                "seed": int(seed),
                "deadline_miss_ratio": float(summary["deadline_miss_ratio"]),
                "mean_waiting_time": float(summary["mean_waiting_time"]),
                "mean_turnaround_time": float(summary["mean_turnaround_time"]),
                "cpu_utilization": float(summary["cpu_utilization"]),
                "assignment_rate": float(summary["assignment_rate"]),
            }
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "episode_ids": episode_ids,
        "seeds": seeds,
        "rl_variant": "hybrid" if bool(args.use_hybrid_rl) else "vanilla",
        "rl_seed_metrics": per_seed,
        "rl_mean_std": {
            "deadline_miss_ratio": _agg([row["deadline_miss_ratio"] for row in per_seed]),
            "mean_waiting_time": _agg([row["mean_waiting_time"] for row in per_seed]),
            "mean_turnaround_time": _agg([row["mean_turnaround_time"] for row in per_seed]),
            "cpu_utilization": _agg([row["cpu_utilization"] for row in per_seed]),
            "assignment_rate": _agg([row["assignment_rate"] for row in per_seed]),
        },
        "baseline_rows": [
            comparison_row_from_summary(scheduler="FCFS", summary=fcfs["summary"]),
            comparison_row_from_summary(scheduler="SJF", summary=sjf["summary"]),
            comparison_row_from_summary(scheduler="RR", summary=rr["summary"]),
        ],
    }

    output_json = args.output_json or (run_dir / "evaluation" / "multi_seed_metrics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Multi-seed evaluation saved: {output_json}")


if __name__ == "__main__":
    main()
