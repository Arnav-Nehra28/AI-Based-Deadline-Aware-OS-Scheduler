import numpy as np
import pandas as pd

try:
    from .pipeline_config import (
        EXPECTED_PROCESSED_COLUMNS,
        MODEL_READY_DATASET_CSV,
        SCALED_DATASET_CSV,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        EXPECTED_PROCESSED_COLUMNS,
        MODEL_READY_DATASET_CSV,
        SCALED_DATASET_CSV,
        ensure_directories,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def main() -> None:
    ensure_directories()

    if not SCALED_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing scaled dataset: {SCALED_DATASET_CSV}. Run scale_resource_features.py first."
        )

    dataset = pd.read_csv(SCALED_DATASET_CSV)

    dataset["cpu_ratio"] = _safe_ratio(dataset["cpu_task"], dataset["cpu_machine"])
    dataset["mem_ratio"] = _safe_ratio(dataset["mem_task"], dataset["mem_machine"])
    dataset["disk_ratio"] = _safe_ratio(dataset["disk_task"], dataset["disk_machine"])

    dataset["cpu_gap"] = dataset["cpu_machine"] - dataset["cpu_task"]
    dataset["mem_gap"] = dataset["mem_machine"] - dataset["mem_task"]

    dataset["resource_pressure"] = (
        dataset["cpu_ratio"] + dataset["mem_ratio"] + dataset["disk_ratio"]
    )
    dataset["task_hour"] = (dataset["start_time"] % 86400) // 3600

    # Keep the processed dataset numerically stable after ratio operations.
    dataset = dataset.replace([np.inf, -np.inf], np.nan).fillna(0)

    for legacy_column in ("instance_id", "end_time"):
        if legacy_column in dataset.columns:
            dataset = dataset.drop(columns=[legacy_column])

    missing_expected_columns = [col for col in EXPECTED_PROCESSED_COLUMNS if col not in dataset.columns]
    if missing_expected_columns:
        raise ValueError(
            "Model-ready dataset is missing expected columns: "
            f"{missing_expected_columns}"
        )

    dataset = dataset[EXPECTED_PROCESSED_COLUMNS]

    before_rows = len(dataset)
    dataset = dataset.drop_duplicates().reset_index(drop=True)
    dropped_duplicates = before_rows - len(dataset)

    dataset.to_csv(MODEL_READY_DATASET_CSV, index=False)

    print(f"Saved model-ready dataset: {MODEL_READY_DATASET_CSV}")
    print(f"Model-ready dataset shape: {dataset.shape}")
    print(f"Dropped duplicate rows: {dropped_duplicates}")


if __name__ == "__main__":
    main()
