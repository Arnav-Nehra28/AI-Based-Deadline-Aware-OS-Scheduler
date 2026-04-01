import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .pipeline_config import PROCESSED_SPLIT_PATHS, ensure_directories
except ImportError:
    from pipeline_config import PROCESSED_SPLIT_PATHS, ensure_directories


def load_processed_splits() -> dict[str, pd.DataFrame]:
    ensure_directories()

    datasets = {}
    missing_files = [path for path in PROCESSED_SPLIT_PATHS.values() if not path.exists()]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Missing processed dataset files: {missing_list}. "
            "Run the preprocessing pipeline before validation or profiling."
        )

    for split_name, path in PROCESSED_SPLIT_PATHS.items():
        datasets[split_name] = pd.read_csv(path)

    return datasets


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    ensure_directories()
    output_path.write_text(json.dumps(to_builtin(report), indent=2), encoding="utf-8")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
