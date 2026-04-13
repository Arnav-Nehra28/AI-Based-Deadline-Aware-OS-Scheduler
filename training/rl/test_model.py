from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from .env_factory import DEFAULT_RL_DATASET_PATH, load_scheduler_dataset
    from .evaluation_core import evaluate_rl_policy_on_episode_ids
except ImportError:
    from env_factory import DEFAULT_RL_DATASET_PATH, load_scheduler_dataset
    from evaluation_core import evaluate_rl_policy_on_episode_ids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained RL model on the held-out test split saved by train_model."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--deadline-slack-factor", type=float, default=2.0)

    parser.add_argument("--top-k-candidates", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-consecutive-defers", type=int, default=30)
    parser.add_argument("--invalid-action-limit", type=int, default=30)
    parser.add_argument("--machine-capacity-scale", type=float, default=1.0)
    parser.add_argument("--machine-pool-size", type=int, default=None)
    return parser


def _load_split_payload(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    split_path = args.split_path or (run_dir / "episode_splits.json")
    model_path = args.model_path or (run_dir / "final_model.zip")
    output_json = args.output_json or (run_dir / "test_metrics.json")

    split_payload = _load_split_payload(split_path)
    test_episode_ids = [int(value) for value in split_payload.get("test_episode_ids", [])]
    if not test_episode_ids:
        raise ValueError(f"No test episode ids found in split file: {split_path}")

    dataset_path = args.dataset or Path(str(split_payload.get("dataset_path", DEFAULT_RL_DATASET_PATH)))
    dataset = load_scheduler_dataset(dataset_path)

    results = evaluate_rl_policy_on_episode_ids(
        model_path=model_path,
        dataset=dataset,
        episode_ids=test_episode_ids,
        seed=int(args.seed),
        device=str(args.device),
        stochastic=bool(args.stochastic),
        deadline_slack_factor=float(args.deadline_slack_factor),
        top_k_candidates=int(args.top_k_candidates),
        max_steps=int(args.max_steps),
        max_consecutive_defers=int(args.max_consecutive_defers),
        invalid_action_limit=int(args.invalid_action_limit),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "test",
        "run_dir": str(run_dir),
        "split_path": str(split_path),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "deadline_slack_factor": float(args.deadline_slack_factor),
        "machine_capacity_scale": float(args.machine_capacity_scale),
        "machine_pool_size": None if args.machine_pool_size is None else int(args.machine_pool_size),
        "results": results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = results["summary"]
    print("Test evaluation complete.")
    print(f"Test episodes: {len(test_episode_ids)}")
    print(f"Deadline miss ratio: {summary['deadline_miss_ratio']:.6f}")
    print(f"Mean waiting time: {summary['mean_waiting_time']}")
    print(f"Mean turnaround time: {summary['mean_turnaround_time']}")
    print(f"CPU utilization: {summary['cpu_utilization']:.6f}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
