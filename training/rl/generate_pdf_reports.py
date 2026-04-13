from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from sb3_contrib import MaskablePPO
except ModuleNotFoundError:
    MaskablePPO = None

try:
    from .env_factory import load_scheduler_dataset
except ImportError:
    from env_factory import load_scheduler_dataset

from rl_pipeline.environment import RewardWeights, TaskSchedulingEnv


DEFAULT_DATASET_PATH = Path("data/interim/rl_env_dataset.json.gz")


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_run_bundle(run_dir: Path) -> dict[str, Any]:
    train_summary = _read_json(run_dir / "train_summary.json")
    experiment_summary = _read_json(run_dir / "experiment_summary.json")
    validation = _read_json(run_dir / "validation_metrics.json")
    test = _read_json(run_dir / "test_metrics.json")
    comparison = _read_json(run_dir / "evaluation" / "comparison_metrics.json")
    results_report = _read_json(run_dir / "results_report.json")
    run_config = _read_json(run_dir / "run_config.json")
    return {
        "train_summary": train_summary,
        "experiment_summary": experiment_summary,
        "validation": validation,
        "test": test,
        "comparison": comparison,
        "results_report": results_report,
        "run_config": run_config,
    }


def _flatten_lines(blocks: list[str], width: int = 96) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        if not block:
            lines.append("")
            continue
        wrapped = textwrap.wrap(block, width=width, break_long_words=False, break_on_hyphens=False)
        lines.extend(wrapped or [""])
        lines.append("")
    return lines


def _add_text_page(pdf: PdfPages, title: str, blocks: list[str], subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")

    fig.text(0.06, 0.965, title, fontsize=20, fontweight="bold", color="#17324d", va="top")
    if subtitle:
        fig.text(0.06, 0.935, subtitle, fontsize=10, color="#5a6b7d", va="top")

    y = 0.90
    for line in _flatten_lines(blocks):
        if y < 0.05:
            break
        fig.text(0.06, y, line, fontsize=10.5, color="#1f2933", va="top", family="monospace" if line.startswith("  ") else None)
        y -= 0.019 if line else 0.011

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_table_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
    *,
    subtitle: str | None = None,
    font_size: float = 10.0,
    scale_y: float = 1.5,
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", color="#17324d", pad=18)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#5a6b7d", va="bottom")

    table = ax.table(
        cellText=dataframe.values,
        colLabels=list(dataframe.columns),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#eef4f8" if row % 2 == 1 else "white")
        cell.set_edgecolor("#c7d3de")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.6f}"


def _comparison_dataframe(comparison_payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in comparison_payload["comparison_table"]:
        rows.append(
            {
                "Scheduler": row["scheduler"],
                "Deadline Miss Ratio": _format_metric(row["deadline_miss_ratio"]),
                "Waiting Time": _format_metric(row["mean_waiting_time"]),
                "Turnaround Time": _format_metric(row["mean_turnaround_time"]),
                "CPU Utilization": _format_metric(row["cpu_utilization"]),
                "Scheduled Tasks": int(row["scheduled_tasks"]),
                "Unscheduled Tasks": int(row["unscheduled_tasks"]),
            }
        )
    return pd.DataFrame(rows)


def _add_comparison_charts(pdf: PdfPages, comparison_payload: dict[str, Any]) -> None:
    frame = pd.DataFrame(comparison_payload["comparison_table"])
    metrics = [
        ("deadline_miss_ratio", "Deadline Miss Ratio", "#c0392b"),
        ("mean_waiting_time", "Waiting Time", "#2980b9"),
        ("mean_turnaround_time", "Turnaround Time", "#16a085"),
        ("cpu_utilization", "CPU Utilization", "#8e44ad"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Scheduler Comparison Charts", fontsize=18, fontweight="bold", color="#17324d")

    for ax, (column, title, color) in zip(axes.flat, metrics):
        sns.barplot(data=frame, x="scheduler", y=column, hue="scheduler", dodge=False, legend=False, palette=[color] * len(frame), ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
        for patch in ax.patches:
            height = patch.get_height()
            ax.annotate(
                f"{height:.4f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 4),
                textcoords="offset points",
            )
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_train_curve_page(pdf: PdfPages, run_dir: Path) -> None:
    eval_npz_path = run_dir / "eval" / "evaluations.npz"
    if not eval_npz_path.exists():
        return

    data = np.load(eval_npz_path)
    timesteps = data["timesteps"]
    rewards = data["results"]
    lengths = data["ep_lengths"]

    reward_mean = rewards.mean(axis=1)
    reward_std = rewards.std(axis=1)
    length_mean = lengths.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Training Progress", fontsize=18, fontweight="bold", color="#17324d")

    axes[0].plot(timesteps, reward_mean, color="#1f77b4", linewidth=2, marker="o")
    axes[0].fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, color="#1f77b4", alpha=0.2)
    axes[0].set_title("Evaluation Reward vs Timesteps", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)

    axes[1].plot(timesteps, length_mean, color="#16a085", linewidth=2, marker="o")
    axes[1].set_title("Evaluation Episode Length vs Timesteps", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Episode Length")
    axes[1].grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_rl_split_metrics_page(pdf: PdfPages, validation_summary: dict[str, Any], test_summary: dict[str, Any]) -> None:
    metric_names = [
        ("deadline_miss_ratio", "DMR"),
        ("mean_waiting_time", "Waiting Time"),
        ("mean_turnaround_time", "Turnaround Time"),
        ("cpu_utilization", "CPU Utilization"),
        ("mean_total_reward", "Mean Reward"),
    ]

    labels = [label for _, label in metric_names]
    val_values = [float(validation_summary[key]) for key, _ in metric_names]
    test_values = [float(test_summary[key]) for key, _ in metric_names]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, val_values, width, label="Validation", color="#2980b9")
    ax.bar(x + width / 2, test_values, width, label="Test", color="#16a085")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("RL Validation vs Test Metrics", fontsize=18, fontweight="bold", color="#17324d")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _deadline_outcome_matrix(comparison_payload: dict[str, Any], baseline_name: str = "SJF") -> np.ndarray:
    rl_eps = comparison_payload["results_by_scheduler"]["RL"]["episodes"]
    baseline_eps = comparison_payload["results_by_scheduler"][baseline_name]["episodes"]

    rl_map: dict[tuple[int, str], bool] = {}
    for episode in rl_eps:
        for row in episode["assignments"]:
            rl_map[(int(row["episode_id"]), str(row["task_id"]))] = bool(row["missed_deadline"])

    baseline_map: dict[tuple[int, str], bool] = {}
    for episode in baseline_eps:
        for row in episode["assignments"]:
            baseline_map[(int(row["episode_id"]), str(row["task_id"]))] = bool(row["missed_deadline"])

    keys = sorted(set(rl_map) | set(baseline_map))
    matrix = np.zeros((2, 2), dtype=int)
    for key in keys:
        baseline_missed = baseline_map.get(key, True)
        rl_missed = rl_map.get(key, True)
        matrix[int(baseline_missed), int(rl_missed)] += 1
    return matrix


def _add_confusion_matrix_page(pdf: PdfPages, comparison_payload: dict[str, Any]) -> None:
    matrix = _deadline_outcome_matrix(comparison_payload, baseline_name="SJF")
    labels = ["Met", "Missed"]

    fig, ax = plt.subplots(figsize=(8.27, 8.27))
    fig.patch.set_facecolor("white")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Deadline Outcome Matrix: SJF vs RL", fontsize=16, fontweight="bold", color="#17324d", pad=16)
    ax.set_xlabel("RL outcome")
    ax.set_ylabel("SJF outcome")

    fig.text(
        0.08,
        0.06,
        "Interpretation: this adapted confusion matrix compares per-task deadline outcomes.\n"
        "Rows show SJF outcomes and columns show RL outcomes on the same test tasks.",
        fontsize=10,
        color="#34495e",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _dataset_profile(dataset_path: Path) -> dict[str, Any]:
    dataset = load_scheduler_dataset(dataset_path)
    task_stats: dict[str, dict[str, float]] = {}
    for column in ["duration", "cpu_demand", "mem_demand", "disk_demand"]:
        series = dataset.tasks[column]
        task_stats[column] = {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
        }

    machine_stats: dict[str, dict[str, float]] = {}
    for column in ["cpu_capacity", "mem_capacity", "disk_capacity"]:
        series = dataset.machines[column]
        machine_stats[column] = {
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
        }

    return {
        "episode_count": int(dataset.tasks["episode_id"].nunique()),
        "task_count": int(len(dataset.tasks)),
        "machine_count": int(len(dataset.machines)),
        "task_stats": task_stats,
        "machine_stats": machine_stats,
    }


def _policy_details(model_path: Path) -> dict[str, Any]:
    if MaskablePPO is None:
        return {"available": False}

    model = MaskablePPO.load(str(model_path), device="cpu")
    policy = model.policy
    return {
        "available": True,
        "policy_class": policy.__class__.__name__,
        "parameter_count": int(sum(param.numel() for param in policy.parameters())),
        "trainable_parameter_count": int(sum(param.numel() for param in policy.parameters() if param.requires_grad)),
        "features_extractor": str(policy.features_extractor),
        "mlp_extractor": str(policy.mlp_extractor),
        "action_head": str(policy.action_net),
        "value_head": str(policy.value_net),
    }


def _relative_gain_lines(comparison_payload: dict[str, Any]) -> list[str]:
    rows = {row["scheduler"]: row for row in comparison_payload["comparison_table"]}
    rl = rows["RL Model"]
    lines: list[str] = []
    baseline_names = [name for name in rows.keys() if name != "RL Model"]
    for baseline_name in baseline_names:
        baseline = rows[baseline_name]
        dmr_delta = float(rl["deadline_miss_ratio"]) - float(baseline["deadline_miss_ratio"])
        wt_delta = float(rl["mean_waiting_time"]) - float(baseline["mean_waiting_time"])
        tat_delta = float(rl["mean_turnaround_time"]) - float(baseline["mean_turnaround_time"])
        cpu_delta = float(rl["cpu_utilization"]) - float(baseline["cpu_utilization"])
        lines.extend(
            [
                f"RL vs {baseline_name}:",
                f"  DMR delta: {dmr_delta:.6f}",
                f"  Waiting-time delta: {wt_delta:.6f}",
                f"  Turnaround-time delta: {tat_delta:.6f}",
                f"  CPU-utilization delta: {cpu_delta:.6f}",
            ]
        )
    return lines


def _build_results_pdf(
    *,
    output_path: Path,
    run_dir: Path,
    bundle: dict[str, Any],
) -> None:
    validation_summary = dict(bundle["validation"]["results"]["summary"])
    test_summary = dict(bundle["test"]["results"]["summary"])
    comparison = bundle["comparison"]
    rows_by_scheduler = {row["scheduler"]: row for row in comparison["comparison_table"]}
    fcfs_dmr = rows_by_scheduler["FCFS"]["deadline_miss_ratio"] if "FCFS" in rows_by_scheduler else None
    sjf_dmr = rows_by_scheduler["SJF"]["deadline_miss_ratio"] if "SJF" in rows_by_scheduler else None
    rr_dmr = rows_by_scheduler["RR"]["deadline_miss_ratio"] if "RR" in rows_by_scheduler else None

    overview_blocks = [
        f"Run directory: {run_dir}",
        f"Dataset: {bundle['results_report']['dataset_path']}",
        f"Device: CUDA GPU-backed training",
        f"Training timesteps: {bundle['train_summary']['total_timesteps']}",
        f"Best mean reward: {bundle['train_summary']['best_mean_reward']}",
        (
            "This report summarizes the final RL scheduler performance, compares RL against "
            "FCFS/SJF/RR, and includes plots plus an adapted confusion matrix for task-level "
            "deadline outcomes."
        ),
        "The headline interpretation should be read from the comparison table and metric-delta section.",
    ]

    with PdfPages(output_path) as pdf:
        _add_text_page(
            pdf,
            "RL Scheduler Results Report",
            overview_blocks,
            subtitle="Detailed performance metrics, comparisons, plots, and evaluation",
        )

        _add_table_page(
            pdf,
            "Final Comparison Table",
            _comparison_dataframe(comparison),
            subtitle="Metrics are computed on the held-out test split.",
        )

        _add_comparison_charts(pdf, comparison)
        _add_train_curve_page(pdf, run_dir)
        _add_rl_split_metrics_page(pdf, validation_summary, test_summary)
        _add_confusion_matrix_page(pdf, comparison)

        evaluation_blocks = [
            "Performance evaluation summary:",
            f"  RL test DMR = {test_summary['deadline_miss_ratio']:.6f}",
            f"  FCFS DMR = {fcfs_dmr:.6f}" if fcfs_dmr is not None else "  FCFS DMR = N/A",
            f"  SJF DMR = {sjf_dmr:.6f}" if sjf_dmr is not None else "  SJF DMR = N/A",
            f"  RR DMR = {rr_dmr:.6f}" if rr_dmr is not None else "  RR DMR = N/A",
            (
                f"  RL waiting time = {test_summary['mean_waiting_time']:.6f}, "
                f"RL turnaround time = {test_summary['mean_turnaround_time']:.6f}"
            ),
            "Interpretation:",
            "  Check metric deltas below to see whether RL beats each baseline on each metric.",
            "",
            "Metric deltas:",
            *_relative_gain_lines(comparison),
        ]
        _add_text_page(pdf, "Evaluation Summary", evaluation_blocks)


def _build_model_pdf(
    *,
    output_path: Path,
    run_dir: Path,
    bundle: dict[str, Any],
    dataset_path: Path,
) -> None:
    dataset_profile = _dataset_profile(dataset_path)
    policy_details = _policy_details(run_dir / "final_model.zip")
    reward_weights = RewardWeights()
    run_args = dict(bundle["run_config"]["args"])

    observation_frame = pd.DataFrame(
        [
            {"Component": "task_features", "Shape": f"{TaskSchedulingEnv.TASK_FEATURE_DIM}"},
            {"Component": "candidate_features", "Shape": f"{run_args['top_k_candidates']} x {TaskSchedulingEnv.CANDIDATE_FEATURE_DIM}"},
            {"Component": "fleet_summary", "Shape": f"{TaskSchedulingEnv.FLEET_SUMMARY_DIM}"},
            {"Component": "action_space", "Shape": f"{run_args['top_k_candidates'] + 1} actions"},
        ]
    )

    reward_frame = pd.DataFrame(
        [
            {"Reward term": "feasible_bonus", "Value": reward_weights.feasible_bonus},
            {"Reward term": "defer_penalty", "Value": reward_weights.defer_penalty},
            {"Reward term": "defer_escalation_rate", "Value": reward_weights.defer_escalation_rate},
            {"Reward term": "invalid_action_penalty", "Value": reward_weights.invalid_action_penalty},
            {"Reward term": "overload_penalty", "Value": reward_weights.overload_penalty},
            {"Reward term": "wait_penalty_weight", "Value": reward_weights.wait_penalty_weight},
            {"Reward term": "missed_deadline_penalty", "Value": reward_weights.missed_deadline_penalty},
            {"Reward term": "lateness_penalty_weight", "Value": reward_weights.lateness_penalty_weight},
            {"Reward term": "balance_bonus_weight", "Value": reward_weights.balance_bonus_weight},
            {"Reward term": "fragmentation_penalty_weight", "Value": reward_weights.fragmentation_penalty_weight},
            {"Reward term": "hotspot_penalty_weight", "Value": reward_weights.hotspot_penalty_weight},
            {"Reward term": "historical_match_bonus", "Value": reward_weights.historical_match_bonus},
            {"Reward term": "completion_bonus", "Value": reward_weights.completion_bonus},
            {"Reward term": "deadline_met_bonus", "Value": reward_weights.deadline_met_bonus},
        ]
    )

    train_config_frame = pd.DataFrame(
        [
            {"Hyperparameter": "device", "Value": run_args["device"]},
            {"Hyperparameter": "total_timesteps", "Value": run_args["total_timesteps"]},
            {"Hyperparameter": "n_envs", "Value": run_args["n_envs"]},
            {"Hyperparameter": "top_k_candidates", "Value": run_args["top_k_candidates"]},
            {"Hyperparameter": "max_steps", "Value": run_args["max_steps"]},
            {"Hyperparameter": "max_consecutive_defers", "Value": run_args["max_consecutive_defers"]},
            {"Hyperparameter": "invalid_action_limit", "Value": run_args["invalid_action_limit"]},
            {"Hyperparameter": "machine_capacity_scale", "Value": run_args["machine_capacity_scale"]},
            {"Hyperparameter": "machine_pool_size", "Value": run_args["machine_pool_size"]},
            {"Hyperparameter": "deadline_slack_factor", "Value": run_args.get("deadline_slack_factor", 2.0)},
            {"Hyperparameter": "learning_rate", "Value": run_args["learning_rate"]},
            {"Hyperparameter": "gamma", "Value": run_args["gamma"]},
            {"Hyperparameter": "gae_lambda", "Value": run_args["gae_lambda"]},
            {"Hyperparameter": "n_steps", "Value": run_args["n_steps"]},
            {"Hyperparameter": "batch_size", "Value": run_args["batch_size"]},
            {"Hyperparameter": "n_epochs", "Value": run_args["n_epochs"]},
            {"Hyperparameter": "clip_range", "Value": run_args["clip_range"]},
            {"Hyperparameter": "ent_coef", "Value": run_args["ent_coef"]},
            {"Hyperparameter": "vf_coef", "Value": run_args["vf_coef"]},
            {"Hyperparameter": "policy_hidden_dims", "Value": run_args["policy_hidden_dims"]},
        ]
    )

    dataset_frame = pd.DataFrame(
        [
            {"Field": "Episodes", "Value": dataset_profile["episode_count"]},
            {"Field": "Tasks", "Value": dataset_profile["task_count"]},
            {"Field": "Machines", "Value": dataset_profile["machine_count"]},
            {"Field": "Task duration mean", "Value": f"{dataset_profile['task_stats']['duration']['mean']:.4f}"},
            {"Field": "Task duration p95", "Value": f"{dataset_profile['task_stats']['duration']['p95']:.4f}"},
            {"Field": "CPU demand mean", "Value": f"{dataset_profile['task_stats']['cpu_demand']['mean']:.4f}"},
            {"Field": "Memory demand mean", "Value": f"{dataset_profile['task_stats']['mem_demand']['mean']:.4f}"},
            {"Field": "CPU capacity mean", "Value": f"{dataset_profile['machine_stats']['cpu_capacity']['mean']:.4f}"},
            {"Field": "Memory capacity mean", "Value": f"{dataset_profile['machine_stats']['mem_capacity']['mean']:.4f}"},
        ]
    )

    model_blocks = [
        "RL formulation:",
        "  State: current task features, candidate machine features, fleet summary.",
        "  Action: choose one of the top-k machine candidates or defer.",
        "  Reward: combines feasible assignment bonus, escalating defer penalties, invalid/overload penalties,",
        "          wait pressure, deadline pressure, terminal completion reward, balance bonus, fragmentation",
        "          penalty, hotspot penalty, and historical match bonus.",
        "",
        "Environment decision logic:",
        "  Infeasible machine actions are masked out before policy selection.",
        "  Pending tasks are sorted by duration (shorter jobs first).",
        "  If the head task is infeasible but another queued task is feasible, that feasible task is promoted.",
        "  When no feasible candidate exists, defer advances simulated time to the next event.",
        "  Episode truncation limit is dynamic: max(base limit, 4 x task count).",
        "",
        "This design lets RL learn machine placement while the environment prevents obviously invalid decisions.",
    ]

    architecture_blocks = [
        f"Policy class: {policy_details.get('policy_class', 'Unavailable')}",
        f"Trainable parameters: {policy_details.get('trainable_parameter_count', 'Unavailable')}",
        f"Total flattened input size: {TaskSchedulingEnv.TASK_FEATURE_DIM + TaskSchedulingEnv.FLEET_SUMMARY_DIM + int(run_args['top_k_candidates']) * TaskSchedulingEnv.CANDIDATE_FEATURE_DIM}",
        "",
        "Feature extraction:",
        f"  {policy_details.get('features_extractor', 'Unavailable')}",
        "",
        "Policy / value network:",
        f"  {policy_details.get('mlp_extractor', 'Unavailable')}",
        "",
        "Action head:",
        f"  {policy_details.get('action_head', 'Unavailable')}",
        "",
        "Value head:",
        f"  {policy_details.get('value_head', 'Unavailable')}",
    ]

    with PdfPages(output_path) as pdf:
        _add_text_page(
            pdf,
            "RL Scheduler Model Details",
            [
                f"Run directory: {run_dir}",
                f"Model artifact: {run_dir / 'final_model.zip'}",
                "This document describes the scheduler formulation, observation/action spaces, training setup,",
                "policy architecture, dataset profile, and the implementation choices behind the best run.",
            ],
            subtitle="Model architecture, environment details, and training configuration",
        )
        _add_text_page(pdf, "Problem Formulation", model_blocks)
        _add_table_page(pdf, "Dataset Profile", dataset_frame, subtitle="Core dataset statistics used by the scheduler environment.")
        _add_table_page(pdf, "Observation and Action Spaces", observation_frame, subtitle="Observation tensor structure consumed by the MaskablePPO policy.")
        _add_table_page(pdf, "Reward Weights", reward_frame, subtitle="Reward shaping terms from the scheduler environment.")
        _add_text_page(pdf, "Policy Architecture", architecture_blocks)
        _add_table_page(pdf, "Training Configuration", train_config_frame, subtitle="Hyperparameters resolved for the fine-tuned best run.", font_size=9.2, scale_y=1.35)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate two PDF reports for an RL run: results/comparison and model-details."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-pdf", type=Path, default=None)
    parser.add_argument("--model-pdf", type=Path, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_pdf = args.results_pdf or (run_dir / "results_performance_comparison_report.pdf")
    model_pdf = args.model_pdf or (run_dir / "model_details_report.pdf")

    bundle = _load_run_bundle(run_dir)

    results_pdf.parent.mkdir(parents=True, exist_ok=True)
    model_pdf.parent.mkdir(parents=True, exist_ok=True)

    _build_results_pdf(output_path=results_pdf, run_dir=run_dir, bundle=bundle)
    _build_model_pdf(output_path=model_pdf, run_dir=run_dir, bundle=bundle, dataset_path=Path(args.dataset))

    print("PDF reports created.")
    print(f"Results PDF: {results_pdf}")
    print(f"Model PDF: {model_pdf}")


if __name__ == "__main__":
    main()
