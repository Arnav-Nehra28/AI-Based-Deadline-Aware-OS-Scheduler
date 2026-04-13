from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a consolidated results file containing training, validation, "
            "test, and scheduler-comparison metrics for one RL run directory."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser


def _build_markdown_report(payload: dict[str, Any]) -> str:
    train = payload["training"]
    val = payload["validation"]
    test = payload["test"]
    rows = payload["comparison_table"]

    lines: list[str] = []
    lines.append("# Consolidated RL Results")
    lines.append("")
    lines.append("## Run Overview")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{payload['generated_at_utc']}`")
    lines.append(f"- Run directory: `{payload['run_dir']}`")
    lines.append(f"- Model path: `{payload['model_path']}`")
    lines.append(f"- Dataset path: `{payload['dataset_path']}`")
    lines.append(f"- Machine capacity scale: `{payload['machine_capacity_scale']}`")
    lines.append(f"- Machine pool size: `{payload['machine_pool_size']}`")
    lines.append("")
    lines.append("## Training Metrics")
    lines.append("")
    lines.append(f"- Total timesteps: `{train['total_timesteps']}`")
    lines.append(f"- Best mean reward: `{train['best_mean_reward']}`")
    lines.append(f"- Train episodes: `{train['train_episode_count']}`")
    lines.append(f"- Validation episodes: `{train['val_episode_count']}`")
    lines.append(f"- Test episodes: `{train['test_episode_count']}`")
    lines.append("")
    lines.append("## Validation Metrics (RL)")
    lines.append("")
    lines.append(f"- Deadline miss ratio: `{val['deadline_miss_ratio']}`")
    lines.append(f"- Mean waiting time: `{val['mean_waiting_time']}`")
    lines.append(f"- Mean turnaround time: `{val['mean_turnaround_time']}`")
    lines.append(f"- CPU utilization: `{val['cpu_utilization']}`")
    lines.append(f"- Mean total reward: `{val['mean_total_reward']}`")
    lines.append(f"- Mean feasible rate: `{val['mean_feasible_rate']}`")
    lines.append(f"- Mean defer rate: `{val['mean_defer_rate']}`")
    lines.append(f"- Mean invalid rate: `{val['mean_invalid_rate']}`")
    lines.append("")
    lines.append("## Test Metrics (RL)")
    lines.append("")
    lines.append(f"- Deadline miss ratio: `{test['deadline_miss_ratio']}`")
    lines.append(f"- Mean waiting time: `{test['mean_waiting_time']}`")
    lines.append(f"- Mean turnaround time: `{test['mean_turnaround_time']}`")
    lines.append(f"- CPU utilization: `{test['cpu_utilization']}`")
    lines.append(f"- Mean total reward: `{test['mean_total_reward']}`")
    lines.append(f"- Mean feasible rate: `{test['mean_feasible_rate']}`")
    lines.append(f"- Mean defer rate: `{test['mean_defer_rate']}`")
    lines.append(f"- Mean invalid rate: `{test['mean_invalid_rate']}`")
    lines.append("")
    lines.append("## Scheduler Comparison")
    lines.append("")
    lines.append("| Scheduler | Deadline Miss Ratio | Waiting Time | Turnaround Time | CPU Utilization |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| "
            f"{row['scheduler']} | "
            f"{row['deadline_miss_ratio']:.6f} | "
            f"{row['mean_waiting_time']} | "
            f"{row['mean_turnaround_time']} | "
            f"{row['cpu_utilization']:.6f} |"
        )
    lines.append("")
    lines.append("## Files Used")
    lines.append("")
    for path in payload["source_files"]:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output_json = args.output_json or (run_dir / "results_report.json")
    output_markdown = args.output_markdown or (run_dir / "results_report.md")

    train_summary_path = run_dir / "train_summary.json"
    experiment_summary_path = run_dir / "experiment_summary.json"
    validation_path = run_dir / "validation_metrics.json"
    test_path = run_dir / "test_metrics.json"
    comparison_path = run_dir / "evaluation" / "comparison_metrics.json"

    train_summary = _read_json(train_summary_path)
    experiment_summary = _read_json(experiment_summary_path)
    validation = _read_json(validation_path)
    test = _read_json(test_path)
    comparison = _read_json(comparison_path)

    val_summary = dict(validation["results"]["summary"])
    test_summary = dict(test["results"]["summary"])

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(train_summary["model_path"]),
        "dataset_path": str(validation["dataset_path"]),
        "deadline_rule": str(val_summary.get("deadline_rule")),
        "deadline_slack_factor": float(validation["deadline_slack_factor"]),
        "machine_capacity_scale": float(validation.get("machine_capacity_scale", 1.0)),
        "machine_pool_size": None
        if validation.get("machine_pool_size") is None
        else int(validation["machine_pool_size"]),
        "training": {
            "total_timesteps": int(train_summary["total_timesteps"]),
            "best_mean_reward": _maybe_float(train_summary.get("best_mean_reward")),
            "train_episode_count": int(experiment_summary["train_episode_count"]),
            "val_episode_count": int(experiment_summary["val_episode_count"]),
            "test_episode_count": int(experiment_summary["test_episode_count"]),
        },
        "validation": {
            "deadline_miss_ratio": float(val_summary["deadline_miss_ratio"]),
            "mean_waiting_time": _maybe_float(val_summary.get("mean_waiting_time")),
            "mean_turnaround_time": _maybe_float(val_summary.get("mean_turnaround_time")),
            "cpu_utilization": float(val_summary["cpu_utilization"]),
            "assignment_rate": float(val_summary["assignment_rate"]),
            "mean_total_reward": _maybe_float(val_summary.get("mean_total_reward")),
            "mean_feasible_rate": _maybe_float(val_summary.get("mean_feasible_rate")),
            "mean_defer_rate": _maybe_float(val_summary.get("mean_defer_rate")),
            "mean_invalid_rate": _maybe_float(val_summary.get("mean_invalid_rate")),
            "terminated_episodes": int(val_summary["terminated_episodes"]),
            "truncated_episodes": int(val_summary["truncated_episodes"]),
        },
        "test": {
            "deadline_miss_ratio": float(test_summary["deadline_miss_ratio"]),
            "mean_waiting_time": _maybe_float(test_summary.get("mean_waiting_time")),
            "mean_turnaround_time": _maybe_float(test_summary.get("mean_turnaround_time")),
            "cpu_utilization": float(test_summary["cpu_utilization"]),
            "assignment_rate": float(test_summary["assignment_rate"]),
            "mean_total_reward": _maybe_float(test_summary.get("mean_total_reward")),
            "mean_feasible_rate": _maybe_float(test_summary.get("mean_feasible_rate")),
            "mean_defer_rate": _maybe_float(test_summary.get("mean_defer_rate")),
            "mean_invalid_rate": _maybe_float(test_summary.get("mean_invalid_rate")),
            "terminated_episodes": int(test_summary["terminated_episodes"]),
            "truncated_episodes": int(test_summary["truncated_episodes"]),
        },
        "comparison_table": list(comparison["comparison_table"]),
        "source_files": [
            str(train_summary_path),
            str(experiment_summary_path),
            str(validation_path),
            str(test_path),
            str(comparison_path),
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = _build_markdown_report(payload)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown, encoding="utf-8")

    print("Consolidated results report created.")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_markdown}")


if __name__ == "__main__":
    main()
