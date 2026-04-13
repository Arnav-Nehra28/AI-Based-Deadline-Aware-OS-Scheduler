from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _metric_order() -> list[tuple[str, str]]:
    return [
        ("deadline_miss_ratio", "Deadline Miss Ratio"),
        ("mean_waiting_time", "Mean Waiting Time"),
        ("mean_turnaround_time", "Mean Turnaround Time"),
        ("cpu_utilization", "CPU Utilization"),
        ("assignment_rate", "Assignment Rate"),
    ]


def _comparison_dataframe(scenario_name: str, payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in payload["comparison_table"]:
        scheduled = int(row["scheduled_tasks"])
        total = int(row["total_tasks"])
        rows.append(
            {
                "Scenario": scenario_name,
                "Scheduler": row["scheduler"],
                "Deadline Miss Ratio": float(row["deadline_miss_ratio"]),
                "Mean Waiting Time": float(row["mean_waiting_time"]),
                "Mean Turnaround Time": float(row["mean_turnaround_time"]),
                "CPU Utilization": float(row["cpu_utilization"]),
                "Scheduled Tasks": scheduled,
                "Unscheduled Tasks": int(row["unscheduled_tasks"]),
                "Assignment Rate": float(scheduled / max(total, 1)),
                "Total Tasks": total,
            }
        )
    return pd.DataFrame(rows)


def _deadline_outcome_matrix(
    payload: dict[str, Any],
    baseline_name: str = "SJF",
) -> np.ndarray:
    rl_episodes = payload["results_by_scheduler"]["RL"]["episodes"]
    baseline_episodes = payload["results_by_scheduler"][baseline_name]["episodes"]

    rl_map: dict[tuple[int, str], bool] = {}
    for episode in rl_episodes:
        for row in episode["assignments"]:
            rl_map[(int(row["episode_id"]), str(row["task_id"]))] = bool(row["missed_deadline"])

    baseline_map: dict[tuple[int, str], bool] = {}
    for episode in baseline_episodes:
        for row in episode["assignments"]:
            baseline_map[(int(row["episode_id"]), str(row["task_id"]))] = bool(row["missed_deadline"])

    keys = sorted(set(rl_map) | set(baseline_map))
    matrix = np.zeros((2, 2), dtype=int)
    for key in keys:
        baseline_missed = baseline_map.get(key, True)
        rl_missed = rl_map.get(key, True)
        matrix[int(baseline_missed), int(rl_missed)] += 1
    return matrix


def _rl_row(frame: pd.DataFrame) -> pd.Series:
    filtered = frame.loc[frame["Scheduler"] == "RL Model"]
    if filtered.empty:
        raise ValueError("RL Model row is missing from comparison table.")
    return filtered.iloc[0]


def _delta_table(frame: pd.DataFrame) -> pd.DataFrame:
    rl = _rl_row(frame)
    rows: list[dict[str, Any]] = []
    for scheduler in ["FCFS", "SJF", "RR"]:
        baseline = frame.loc[frame["Scheduler"] == scheduler]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        rows.append(
            {
                "Baseline": scheduler,
                "DMR Delta (RL - baseline)": float(rl["Deadline Miss Ratio"] - base["Deadline Miss Ratio"]),
                "Wait Delta (RL - baseline)": float(rl["Mean Waiting Time"] - base["Mean Waiting Time"]),
                "TAT Delta (RL - baseline)": float(rl["Mean Turnaround Time"] - base["Mean Turnaround Time"]),
                "CPU Delta (RL - baseline)": float(rl["CPU Utilization"] - base["CPU Utilization"]),
                "Assignment Delta (RL - baseline)": float(rl["Assignment Rate"] - base["Assignment Rate"]),
            }
        )
    return pd.DataFrame(rows)


def _win_matrix(stress_frame: pd.DataFrame, medium_frame: pd.DataFrame) -> pd.DataFrame:
    scenarios = [("Stress", stress_frame), ("Medium", medium_frame)]
    rows: list[dict[str, Any]] = []
    for scenario_name, frame in scenarios:
        rl = _rl_row(frame)
        for scheduler in ["FCFS", "SJF", "RR"]:
            baseline = frame.loc[frame["Scheduler"] == scheduler]
            if baseline.empty:
                continue
            base = baseline.iloc[0]
            rows.append(
                {
                    "Scenario": scenario_name,
                    "Baseline": scheduler,
                    "DMR Better": int(rl["Deadline Miss Ratio"] < base["Deadline Miss Ratio"]),
                    "Wait Better": int(rl["Mean Waiting Time"] < base["Mean Waiting Time"]),
                    "TAT Better": int(rl["Mean Turnaround Time"] < base["Mean Turnaround Time"]),
                    "CPU Better": int(rl["CPU Utilization"] > base["CPU Utilization"]),
                    "Assign Better": int(rl["Assignment Rate"] > base["Assignment Rate"]),
                }
            )
    matrix = pd.DataFrame(rows)
    matrix["Wins"] = (
        matrix["DMR Better"]
        + matrix["Wait Better"]
        + matrix["TAT Better"]
        + matrix["CPU Better"]
        + matrix["Assign Better"]
    )
    return matrix


def _save_metric_bar_charts(combined: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "metric_bars_by_scenario.png"
    metrics = _metric_order()
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    axes_flat = axes.flatten()
    for axis, (column, title) in zip(axes_flat, metrics):
        sns.barplot(
            data=combined,
            x="Scheduler",
            y=title,
            hue="Scenario",
            palette={"Stress": "#c0392b", "Medium": "#2980b9"},
            ax=axis,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlabel("")
        if axis.legend_ is not None:
            axis.legend(loc="best", fontsize=8)
    # Remove unused subplot.
    for axis in axes_flat[len(metrics):]:
        axis.axis("off")
    fig.suptitle("Scheduler Metrics Across Stress and Medium", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_assignment_chart(combined: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "assignment_and_unscheduled.png"
    chart_frame = combined.copy()
    chart_frame["Assignment %"] = chart_frame["Assignment Rate"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=chart_frame,
        x="Scheduler",
        y="Assignment %",
        hue="Scenario",
        palette={"Stress": "#c0392b", "Medium": "#2980b9"},
        ax=axes[0],
    )
    axes[0].set_title("Assignment Rate (%)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_xlabel("")

    sns.barplot(
        data=chart_frame,
        x="Scheduler",
        y="Unscheduled Tasks",
        hue="Scenario",
        palette={"Stress": "#e67e22", "Medium": "#2ecc71"},
        ax=axes[1],
    )
    axes[1].set_title("Unscheduled Tasks")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_xlabel("")
    if axes[1].legend_ is not None:
        axes[1].legend(loc="best", fontsize=8)
    if axes[0].legend_ is not None:
        axes[0].legend(loc="best", fontsize=8)

    fig.suptitle("Task Completion Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_training_curve(run_dir: Path, output_dir: Path) -> Path | None:
    eval_npz = run_dir / "eval" / "evaluations.npz"
    if not eval_npz.exists():
        return None
    data = np.load(eval_npz)
    timesteps = data["timesteps"]
    rewards = data["results"]
    ep_lengths = data["ep_lengths"]
    reward_mean = rewards.mean(axis=1)
    reward_std = rewards.std(axis=1)
    lengths_mean = ep_lengths.mean(axis=1)

    output_path = output_dir / "training_progress_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(timesteps, reward_mean, color="#1f77b4", linewidth=2)
    axes[0].fill_between(timesteps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color="#1f77b4")
    axes[0].set_title("Evaluation Reward vs Timesteps")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.2)

    axes[1].plot(timesteps, lengths_mean, color="#16a085", linewidth=2)
    axes[1].set_title("Evaluation Episode Length vs Timesteps")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Episode Length")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_confusion_matrix_plot(
    stress_payload: dict[str, Any],
    medium_payload: dict[str, Any],
    output_dir: Path,
) -> Path:
    output_path = output_dir / "deadline_outcome_matrices_sjf_vs_rl.png"
    stress_matrix = _deadline_outcome_matrix(stress_payload, baseline_name="SJF")
    medium_matrix = _deadline_outcome_matrix(medium_payload, baseline_name="SJF")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Met", "Missed"]
    sns.heatmap(stress_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Stress: SJF vs RL")
    axes[0].set_xlabel("RL Outcome")
    axes[0].set_ylabel("SJF Outcome")

    sns.heatmap(medium_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Medium: SJF vs RL")
    axes[1].set_xlabel("RL Outcome")
    axes[1].set_ylabel("SJF Outcome")

    fig.suptitle("Deadline Outcome Matrices", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_win_heatmap(win_matrix: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "rl_win_matrix_heatmap.png"
    plot_frame = win_matrix.copy()
    plot_frame["Pair"] = plot_frame["Scenario"] + " vs " + plot_frame["Baseline"]
    value_columns = ["DMR Better", "Wait Better", "TAT Better", "CPU Better", "Assign Better"]
    heatmap_frame = plot_frame.set_index("Pair")[value_columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_frame, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=0.5, ax=ax)
    ax.set_title("RL Win Matrix (1 = RL better, 0 = baseline better)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Scenario/Baseline Pair")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _add_table_page(pdf: PdfPages, title: str, dataframe: pd.DataFrame, subtitle: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=16, fontweight="bold", color="#17324d", pad=16)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#5a6b7d")

    table = ax.table(
        cellText=dataframe.values,
        colLabels=list(dataframe.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.4)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#eef4f8" if row % 2 == 1 else "white")
            cell.set_edgecolor("#d0d9e2")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")
    img = plt.imread(image_path)
    ax.imshow(img)
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#17324d")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    output_path: Path,
    stress_frame: pd.DataFrame,
    medium_frame: pd.DataFrame,
    stress_delta: pd.DataFrame,
    medium_delta: pd.DataFrame,
    win_matrix: pd.DataFrame,
    plots: list[Path],
) -> None:
    def table_md(df: pd.DataFrame) -> str:
        columns = [str(column) for column in df.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in df.itertuples(index=False):
            values = [str(value) for value in row]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append("# RL Performance Metrics File")
    lines.append("")
    lines.append("## Stress Comparison")
    lines.append("")
    lines.append(table_md(stress_frame))
    lines.append("")
    lines.append("## Medium Comparison")
    lines.append("")
    lines.append(table_md(medium_frame))
    lines.append("")
    lines.append("## RL vs Baselines Delta (Stress)")
    lines.append("")
    lines.append(table_md(stress_delta))
    lines.append("")
    lines.append("## RL vs Baselines Delta (Medium)")
    lines.append("")
    lines.append(table_md(medium_delta))
    lines.append("")
    lines.append("## RL Win Matrix")
    lines.append("")
    lines.append(table_md(win_matrix))
    lines.append("")
    lines.append("## Graphs and Matrices")
    lines.append("")
    for plot in plots:
        rel = os.path.relpath(plot, start=output_path.parent).replace("\\", "/")
        lines.append(f"### {plot.stem}")
        lines.append("")
        lines.append(f"![{plot.stem}]({rel})")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_payload(
    *,
    run_dir: Path,
    stress_payload: dict[str, Any],
    medium_payload: dict[str, Any],
    stress_frame: pd.DataFrame,
    medium_frame: pd.DataFrame,
    stress_delta: pd.DataFrame,
    medium_delta: pd.DataFrame,
    win_matrix: pd.DataFrame,
    plots: list[Path],
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "stress_metrics_path": str(run_dir / "evaluation" / "comparison_metrics_stress_final_with_rr.json"),
        "medium_metrics_path": str(run_dir / "evaluation" / "comparison_metrics_medium_final_with_rr.json"),
        "stress_comparison_table": json.loads(stress_frame.to_json(orient="records")),
        "medium_comparison_table": json.loads(medium_frame.to_json(orient="records")),
        "stress_rl_vs_baseline_delta": json.loads(stress_delta.to_json(orient="records")),
        "medium_rl_vs_baseline_delta": json.loads(medium_delta.to_json(orient="records")),
        "rl_win_matrix": json.loads(win_matrix.to_json(orient="records")),
        "stress_deadline_outcome_matrix_sjf_vs_rl": _deadline_outcome_matrix(stress_payload, "SJF").tolist(),
        "medium_deadline_outcome_matrix_sjf_vs_rl": _deadline_outcome_matrix(medium_payload, "SJF").tolist(),
        "plot_files": [str(path) for path in plots],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one comprehensive performance metrics file for RL scheduling, "
            "including comparisons, graphs, and matrix views across stress and medium scenarios."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stress-json", type=Path, default=None)
    parser.add_argument("--medium-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    evaluation_dir = run_dir / "evaluation"

    stress_json = args.stress_json or (evaluation_dir / "comparison_metrics_stress_final_with_rr.json")
    medium_json = args.medium_json or (evaluation_dir / "comparison_metrics_medium_final_with_rr.json")
    output_json = args.output_json or (evaluation_dir / "performance_metrics_full.json")
    output_markdown = args.output_markdown or (evaluation_dir / "performance_metrics_full.md")
    output_pdf = args.output_pdf or (evaluation_dir / "performance_metrics_full_report.pdf")
    plots_dir = args.plots_dir or (evaluation_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    stress_payload = _read_json(stress_json)
    medium_payload = _read_json(medium_json)

    stress_frame = _comparison_dataframe("Stress", stress_payload)
    medium_frame = _comparison_dataframe("Medium", medium_payload)
    stress_delta = _delta_table(stress_frame)
    medium_delta = _delta_table(medium_frame)
    win_matrix = _win_matrix(stress_frame, medium_frame)
    combined = pd.concat([stress_frame, medium_frame], ignore_index=True)

    metric_bars = _save_metric_bar_charts(combined, plots_dir)
    assignment_plot = _save_assignment_chart(combined, plots_dir)
    confusion_plot = _save_confusion_matrix_plot(stress_payload, medium_payload, plots_dir)
    win_heatmap = _save_win_heatmap(win_matrix, plots_dir)
    training_curve = _save_training_curve(run_dir, plots_dir)

    plot_paths = [metric_bars, assignment_plot, confusion_plot, win_heatmap]
    if training_curve is not None:
        plot_paths.append(training_curve)

    payload = _build_payload(
        run_dir=run_dir,
        stress_payload=stress_payload,
        medium_payload=medium_payload,
        stress_frame=stress_frame,
        medium_frame=medium_frame,
        stress_delta=stress_delta,
        medium_delta=medium_delta,
        win_matrix=win_matrix,
        plots=plot_paths,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _write_markdown(
        output_path=output_markdown,
        stress_frame=stress_frame,
        medium_frame=medium_frame,
        stress_delta=stress_delta,
        medium_delta=medium_delta,
        win_matrix=win_matrix,
        plots=plot_paths,
    )

    with PdfPages(output_pdf) as pdf:
        _add_table_page(
            pdf,
            "Performance Metrics Summary (Stress)",
            stress_frame.round(6),
            subtitle="Schedulers compared on stress scenario.",
        )
        _add_table_page(
            pdf,
            "Performance Metrics Summary (Medium)",
            medium_frame.round(6),
            subtitle="Schedulers compared on medium scenario.",
        )
        _add_table_page(
            pdf,
            "RL vs Baselines Delta (Stress)",
            stress_delta.round(6),
        )
        _add_table_page(
            pdf,
            "RL vs Baselines Delta (Medium)",
            medium_delta.round(6),
        )
        _add_table_page(
            pdf,
            "RL Win Matrix",
            win_matrix,
            subtitle="1 means RL is better than the baseline for that metric.",
        )
        _add_image_page(pdf, metric_bars, "Metric Graphs by Scenario")
        _add_image_page(pdf, assignment_plot, "Assignment and Unscheduled Comparison")
        _add_image_page(pdf, confusion_plot, "Deadline Outcome Matrices (SJF vs RL)")
        _add_image_page(pdf, win_heatmap, "RL Win Matrix Heatmap")
        if training_curve is not None:
            _add_image_page(pdf, training_curve, "Training Curves")

    print("Comprehensive performance metrics files created.")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_markdown}")
    print(f"PDF: {output_pdf}")
    print(f"Plots dir: {plots_dir}")


if __name__ == "__main__":
    main()
