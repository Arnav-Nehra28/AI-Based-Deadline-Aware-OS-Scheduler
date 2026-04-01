import math

import pandas as pd

try:
    from .pipeline_config import (
        EXPECTED_PROCESSED_COLUMNS,
        NUMERIC_PROCESSED_COLUMNS,
        RATIO_COLUMNS,
        SCALED_FEATURE_COLUMNS,
        VALIDATION_REPORT_JSON,
        ensure_directories,
    )
    from .processed_data_utils import load_processed_splits, write_json_report
except ImportError:
    from pipeline_config import (
        EXPECTED_PROCESSED_COLUMNS,
        NUMERIC_PROCESSED_COLUMNS,
        RATIO_COLUMNS,
        SCALED_FEATURE_COLUMNS,
        VALIDATION_REPORT_JSON,
        ensure_directories,
    )
    from processed_data_utils import load_processed_splits, write_json_report


def validate_split(split_name: str, df: pd.DataFrame) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [col for col in EXPECTED_PROCESSED_COLUMNS if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in EXPECTED_PROCESSED_COLUMNS]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    if extra_columns:
        warnings.append(f"Unexpected extra columns: {extra_columns}")

    null_count = int(df.isna().sum().sum())
    if null_count:
        errors.append(f"Found {null_count} null values.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicate rows.")

    empty_machine_ids = int(df["machine_id"].astype(str).str.strip().eq("").sum())
    if empty_machine_ids:
        errors.append(f"Found {empty_machine_ids} empty machine_id values.")

    for column in NUMERIC_PROCESSED_COLUMNS:
        if column not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(f"Column '{column}' is not numeric.")
            continue

        finite_mask = df[column].map(math.isfinite)
        non_finite_count = int((~finite_mask).sum())
        if non_finite_count:
            errors.append(f"Column '{column}' contains {non_finite_count} non-finite values.")

    if "duration" in df.columns:
        invalid_duration_count = int((df["duration"] <= 0).sum())
        if invalid_duration_count:
            errors.append(f"Found {invalid_duration_count} rows with non-positive duration.")

    if "start_time" in df.columns:
        negative_start_count = int((df["start_time"] < 0).sum())
        if negative_start_count:
            errors.append(f"Found {negative_start_count} rows with negative start_time.")

    if "task_hour" in df.columns:
        invalid_task_hour_count = int(((df["task_hour"] < 0) | (df["task_hour"] > 23)).sum())
        if invalid_task_hour_count:
            errors.append(f"Found {invalid_task_hour_count} rows with invalid task_hour values.")

    for column in SCALED_FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        below_zero = int((df[column] < -1e-9).sum())
        above_one = int((df[column] > 1 + 1e-9).sum())
        if below_zero or above_one:
            errors.append(
                f"Scaled column '{column}' is outside [0, 1] for {below_zero + above_one} rows."
            )

    for column in RATIO_COLUMNS:
        if column not in df.columns:
            continue
        negative_count = int((df[column] < 0).sum())
        if negative_count:
            errors.append(f"Ratio column '{column}' has {negative_count} negative values.")

    constant_columns = [
        column
        for column in df.columns
        if int(df[column].nunique(dropna=False)) <= 1
    ]
    if constant_columns:
        warnings.append(f"Constant columns detected: {constant_columns}")

    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "null_count": null_count,
        "duplicate_count": duplicate_count,
        "machine_id_unique": int(df["machine_id"].nunique()) if "machine_id" in df.columns else 0,
        "duration_min": float(df["duration"].min()) if "duration" in df.columns else None,
        "duration_max": float(df["duration"].max()) if "duration" in df.columns else None,
    }

    return {
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


def main(raise_on_error: bool = True) -> dict[str, object]:
    ensure_directories()
    datasets = load_processed_splits()

    report: dict[str, object] = {"splits": {}, "global_warnings": [], "status": "passed"}

    for split_name, df in datasets.items():
        report["splits"][split_name] = validate_split(split_name, df)

    train_machine_ids = set(datasets["train"]["machine_id"])
    for split_name in ("validation", "test"):
        unseen = sorted(set(datasets[split_name]["machine_id"]) - train_machine_ids)
        unseen_count = len(unseen)
        if unseen_count:
            report["global_warnings"].append(
                f"{split_name} contains {unseen_count} machine_id values not present in train."
            )

    has_errors = any(report["splits"][name]["errors"] for name in report["splits"])
    report["status"] = "failed" if has_errors else "passed"

    write_json_report(report, VALIDATION_REPORT_JSON)

    print(f"Validation status: {report['status']}")
    print(f"Saved validation report: {VALIDATION_REPORT_JSON}")

    for split_name, split_report in report["splits"].items():
        print(
            f"{split_name}: errors={len(split_report['errors'])}, "
            f"warnings={len(split_report['warnings'])}"
        )

    for warning in report["global_warnings"]:
        print(f"warning: {warning}")

    if has_errors and raise_on_error:
        raise SystemExit(1)

    return report


if __name__ == "__main__":
    main()
