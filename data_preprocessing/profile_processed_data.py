from statistics import mean

try:
    from .pipeline_config import (
        EXPECTED_PROCESSED_COLUMNS,
        GAP_COLUMNS,
        PROFILE_REPORT_JSON,
        PROFILE_REPORT_MD,
        RATIO_COLUMNS,
        SCALED_FEATURE_COLUMNS,
        ensure_directories,
    )
    from .processed_data_utils import load_processed_splits, write_json_report
except ImportError:
    from pipeline_config import (
        EXPECTED_PROCESSED_COLUMNS,
        GAP_COLUMNS,
        PROFILE_REPORT_JSON,
        PROFILE_REPORT_MD,
        RATIO_COLUMNS,
        SCALED_FEATURE_COLUMNS,
        ensure_directories,
    )
    from processed_data_utils import load_processed_splits, write_json_report


PROFILE_COLUMNS = [
    "duration",
    "start_time",
    *SCALED_FEATURE_COLUMNS,
    *RATIO_COLUMNS,
    *GAP_COLUMNS,
    "resource_pressure",
    "task_hour",
]


def summarize_numeric_column(series):
    quantiles = series.quantile([0.5, 0.9, 0.99])
    return {
        "min": float(series.min()),
        "mean": float(series.mean()),
        "median": float(quantiles.loc[0.5]),
        "p90": float(quantiles.loc[0.9]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(series.max()),
    }


def build_profile() -> dict[str, object]:
    datasets = load_processed_splits()
    train_machine_ids = set(datasets["train"]["machine_id"])

    report: dict[str, object] = {
        "dataset_contract": {
            "expected_columns": EXPECTED_PROCESSED_COLUMNS,
            "target_columns": ["machine_id", "duration"],
            "derived_feature_columns": [
                "cpu_ratio",
                "mem_ratio",
                "disk_ratio",
                "cpu_gap",
                "mem_gap",
                "resource_pressure",
                "task_hour",
            ],
        },
        "splits": {},
        "cross_split_summary": {},
    }

    for split_name, df in datasets.items():
        split_profile = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "machine_id_unique": int(df["machine_id"].nunique()),
            "null_count": int(df.isna().sum().sum()),
            "duplicate_count": int(df.duplicated().sum()),
            "constant_columns": [
                column for column in df.columns if int(df[column].nunique(dropna=False)) <= 1
            ],
            "numeric_summary": {
                column: summarize_numeric_column(df[column])
                for column in PROFILE_COLUMNS
                if column in df.columns
            },
            "task_hour_distribution": {
                str(hour): int(count)
                for hour, count in df["task_hour"].value_counts().sort_index().items()
            },
        }

        if split_name == "train":
            split_profile["machine_id_overlap_with_train"] = 1.0
            split_profile["unseen_machine_ids_vs_train"] = 0
        else:
            current_machine_ids = set(df["machine_id"])
            split_profile["machine_id_overlap_with_train"] = float(
                len(current_machine_ids & train_machine_ids) / max(len(current_machine_ids), 1)
            )
            split_profile["unseen_machine_ids_vs_train"] = int(
                len(current_machine_ids - train_machine_ids)
            )

        report["splits"][split_name] = split_profile

    duration_medians = [
        report["splits"][split_name]["numeric_summary"]["duration"]["median"]
        for split_name in report["splits"]
    ]
    report["cross_split_summary"] = {
        "split_row_counts": {
            split_name: report["splits"][split_name]["shape"][0]
            for split_name in report["splits"]
        },
        "average_duplicate_rows_per_split": float(
            mean(report["splits"][split_name]["duplicate_count"] for split_name in report["splits"])
        ),
        "duration_median_range": [float(min(duration_medians)), float(max(duration_medians))],
    }

    return report


def write_markdown_report(report: dict[str, object]) -> None:
    lines = [
        "# Processed Dataset Profile",
        "",
        "## Dataset Contract",
        "",
        f"- Expected columns: `{', '.join(report['dataset_contract']['expected_columns'])}`",
        f"- Target columns: `{', '.join(report['dataset_contract']['target_columns'])}`",
        (
            "- Derived feature columns: "
            f"`{', '.join(report['dataset_contract']['derived_feature_columns'])}`"
        ),
        "",
        "## Split Summary",
        "",
        "| Split | Rows | Columns | Unique machine_id | Duplicate rows | Unseen machine_id vs train | Constant columns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for split_name, split_report in report["splits"].items():
        constant_columns = ", ".join(split_report["constant_columns"]) or "-"
        lines.append(
            f"| {split_name} | {split_report['shape'][0]} | {split_report['shape'][1]} | "
            f"{split_report['machine_id_unique']} | {split_report['duplicate_count']} | "
            f"{split_report['unseen_machine_ids_vs_train']} | {constant_columns} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                f"- Average duplicate rows per split: "
                f"{report['cross_split_summary']['average_duplicate_rows_per_split']:.2f}"
            ),
            (
                "- Duration median range across splits: "
                f"{report['cross_split_summary']['duration_median_range'][0]:.2f} to "
                f"{report['cross_split_summary']['duration_median_range'][1]:.2f}"
            ),
        ]
    )

    PROFILE_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, object]:
    ensure_directories()
    report = build_profile()
    write_json_report(report, PROFILE_REPORT_JSON)
    write_markdown_report(report)

    print(f"Saved profile report: {PROFILE_REPORT_JSON}")
    print(f"Saved markdown summary: {PROFILE_REPORT_MD}")

    for split_name, split_report in report["splits"].items():
        print(
            f"{split_name}: rows={split_report['shape'][0]}, "
            f"unique_machine_id={split_report['machine_id_unique']}, "
            f"duplicates={split_report['duplicate_count']}"
        )

    return report


if __name__ == "__main__":
    main()
