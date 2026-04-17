from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

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


def _parse_episode_ids(raw_value: str | None) -> list[int]:
    if raw_value is None or not raw_value.strip():
        return []
    values: list[int] = []
    for chunk in raw_value.split(","):
        value = chunk.strip()
        if value:
            values.append(int(value))
    return sorted(set(values))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run final evaluation and comparison for FCFS, SJF, RR, and RL model outputs "
            "on the test split (or user-provided episode ids)."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--episode-ids", type=str, default=None)

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
    parser.add_argument("--include-hybrid-rl", action="store_true")
    parser.add_argument("--hybrid-defer-wait-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--hybrid-high-utilization-threshold", type=float, default=0.90)

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser


def _load_split_payload(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Scheduler | Deadline Miss Ratio | Waiting Time | Turnaround Time | CPU Utilization | Assignment Rate | Scheduled / Total |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        total_tasks = int(row["total_tasks"])
        scheduled_tasks = int(row["scheduled_tasks"])
        assignment_rate = float(scheduled_tasks / max(total_tasks, 1))
        lines.append(
            "| "
            f"{row['scheduler']} | "
            f"{row['deadline_miss_ratio']:.6f} | "
            f"{row['mean_waiting_time']} | "
            f"{row['mean_turnaround_time']} | "
            f"{row['cpu_utilization']:.6f} | "
            f"{assignment_rate:.6f} | "
            f"{scheduled_tasks}/{total_tasks} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    split_path = args.split_path or (run_dir / "episode_splits.json")
    split_payload = _load_split_payload(split_path)

    dataset_path = args.dataset or Path(str(split_payload.get("dataset_path", DEFAULT_RL_DATASET_PATH)))
    model_path = args.model_path or (run_dir / "final_model.zip")

    requested_ids = _parse_episode_ids(args.episode_ids)
    if requested_ids:
        episode_ids = requested_ids
    else:
        episode_ids = [int(value) for value in split_payload.get("test_episode_ids", [])]
    if not episode_ids:
        raise ValueError("No episode ids provided and no test episode ids found in split file.")

    output_root = run_dir / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (output_root / "comparison_metrics.json")
    output_csv = args.output_csv or (output_root / "comparison_table.csv")
    output_markdown = args.output_markdown or (output_root / "comparison_table.md")

    dataset = load_scheduler_dataset(dataset_path)

    rl_results = evaluate_rl_policy_on_episode_ids(
        model_path=model_path,
        dataset=dataset,
        episode_ids=episode_ids,
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
        use_hybrid_scheduler=False,
        hybrid_defer_wait_ratio_threshold=float(args.hybrid_defer_wait_ratio_threshold),
        hybrid_high_utilization_threshold=float(args.hybrid_high_utilization_threshold),
    )
    fcfs_results = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset,
        episode_ids=episode_ids,
        heuristic="FCFS",
        deadline_slack_factor=float(args.deadline_slack_factor),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )
    sjf_results = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset,
        episode_ids=episode_ids,
        heuristic="SJF",
        deadline_slack_factor=float(args.deadline_slack_factor),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )
    rr_results = evaluate_heuristic_policy_on_episode_ids(
        dataset=dataset,
        episode_ids=episode_ids,
        heuristic="RR",
        deadline_slack_factor=float(args.deadline_slack_factor),
        machine_capacity_scale=float(args.machine_capacity_scale),
        machine_pool_size=None if args.machine_pool_size is None else int(args.machine_pool_size),
    )

    comparison_rows = [
        comparison_row_from_summary(scheduler="FCFS", summary=fcfs_results["summary"]),
        comparison_row_from_summary(scheduler="SJF", summary=sjf_results["summary"]),
        comparison_row_from_summary(scheduler="RR", summary=rr_results["summary"]),
        comparison_row_from_summary(scheduler="RL Model", summary=rl_results["summary"]),
    ]
    hybrid_results = None
    if bool(args.include_hybrid_rl):
        hybrid_results = evaluate_rl_policy_on_episode_ids(
            model_path=model_path,
            dataset=dataset,
            episode_ids=episode_ids,
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
            use_hybrid_scheduler=True,
            hybrid_defer_wait_ratio_threshold=float(args.hybrid_defer_wait_ratio_threshold),
            hybrid_high_utilization_threshold=float(args.hybrid_high_utilization_threshold),
        )
        comparison_rows.append(
            comparison_row_from_summary(scheduler="RL Hybrid", summary=hybrid_results["summary"])
        )

    results_by_scheduler = {
        "FCFS": fcfs_results,
        "SJF": sjf_results,
        "RR": rr_results,
        "RL": rl_results,
    }
    if hybrid_results is not None:
        results_by_scheduler["RL_HYBRID"] = hybrid_results

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "split_path": str(split_path),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "deadline_slack_factor": float(args.deadline_slack_factor),
        "machine_capacity_scale": float(args.machine_capacity_scale),
        "machine_pool_size": None if args.machine_pool_size is None else int(args.machine_pool_size),
        "include_hybrid_rl": bool(args.include_hybrid_rl),
        "hybrid_defer_wait_ratio_threshold": float(args.hybrid_defer_wait_ratio_threshold),
        "hybrid_high_utilization_threshold": float(args.hybrid_high_utilization_threshold),
        "episode_ids": episode_ids,
        "comparison_table": comparison_rows,
        "results_by_scheduler": results_by_scheduler,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "scheduler",
                "deadline_miss_ratio",
                "mean_waiting_time",
                "mean_turnaround_time",
                "cpu_utilization",
                "total_tasks",
                "scheduled_tasks",
                "unscheduled_tasks",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    _write_markdown_table(output_markdown, comparison_rows)

    print("Final evaluation complete.")
    print(f"Episode count: {len(episode_ids)}")
    print("")
    print("| Scheduler | Deadline Miss Ratio | Waiting Time | Turnaround Time | CPU Utilization | Assignment Rate | Scheduled / Total |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in comparison_rows:
        total_tasks = int(row["total_tasks"])
        scheduled_tasks = int(row["scheduled_tasks"])
        assignment_rate = float(scheduled_tasks / max(total_tasks, 1))
        print(
            "| "
            f"{row['scheduler']} | "
            f"{row['deadline_miss_ratio']:.6f} | "
            f"{row['mean_waiting_time']} | "
            f"{row['mean_turnaround_time']} | "
            f"{row['cpu_utilization']:.6f} | "
            f"{assignment_rate:.6f} | "
            f"{scheduled_tasks}/{total_tasks} |"
        )
    print("")
    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved Markdown: {output_markdown}")


if __name__ == "__main__":
    main()
