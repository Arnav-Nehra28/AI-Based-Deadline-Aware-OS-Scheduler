from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TASK_COLUMNS = [
    "episode_id",
    "task_index",
    "task_id",
    "arrival_time",
    "duration",
    "cpu_demand",
    "mem_demand",
    "disk_demand",
    "historical_machine_id",
]

MACHINE_COLUMNS = ["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]
EPISODE_COLUMNS = ["episode_id", "task_count", "start_time", "end_time", "source_kind"]


@dataclass(frozen=True)
class RLEnvDataset:
    tasks: pd.DataFrame
    machines: pd.DataFrame
    episodes: pd.DataFrame
    metadata: dict[str, Any]


def _ensure_columns(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in required_columns:
        if column not in output.columns:
            output[column] = pd.Series(dtype="object")
    return output[required_columns]


def validate_env_dataset(dataset: RLEnvDataset) -> None:
    tasks = _ensure_columns(dataset.tasks, TASK_COLUMNS)
    machines = _ensure_columns(dataset.machines, MACHINE_COLUMNS)
    episodes = _ensure_columns(dataset.episodes, EPISODE_COLUMNS)

    if tasks.empty:
        raise ValueError("RL environment dataset has no tasks.")
    if machines.empty:
        raise ValueError("RL environment dataset has no machines.")
    if episodes.empty:
        raise ValueError("RL environment dataset has no episodes.")

    if tasks["episode_id"].isnull().any():
        raise ValueError("All tasks must belong to an episode.")
    if tasks["task_index"].isnull().any():
        raise ValueError("All tasks must have a task_index within their episode.")

    episode_ids = set(episodes["episode_id"].tolist())
    if not set(tasks["episode_id"].tolist()).issubset(episode_ids):
        raise ValueError("Tasks reference episode_ids that are missing from the episodes table.")

    machine_ids = set(machines["machine_id"].astype(str).tolist())
    historical_ids = set(tasks["historical_machine_id"].dropna().astype(str).tolist())
    if not historical_ids.issubset(machine_ids):
        raise ValueError("Tasks reference historical machines that are missing from the machine catalog.")


def save_env_dataset(dataset: RLEnvDataset, output_path: Path) -> None:
    validate_env_dataset(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": dataset.metadata,
        "tasks": _ensure_columns(dataset.tasks, TASK_COLUMNS).to_dict(orient="records"),
        "machines": _ensure_columns(dataset.machines, MACHINE_COLUMNS).to_dict(orient="records"),
        "episodes": _ensure_columns(dataset.episodes, EPISODE_COLUMNS).to_dict(orient="records"),
    }

    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj)
    else:
        output_path.write_text(json.dumps(payload), encoding="utf-8")


def load_env_dataset(input_path: str | Path) -> RLEnvDataset:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing RL environment dataset artifact: {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    dataset = RLEnvDataset(
        tasks=_ensure_columns(pd.DataFrame(payload.get("tasks", [])), TASK_COLUMNS),
        machines=_ensure_columns(pd.DataFrame(payload.get("machines", [])), MACHINE_COLUMNS),
        episodes=_ensure_columns(pd.DataFrame(payload.get("episodes", [])), EPISODE_COLUMNS),
        metadata=dict(payload.get("metadata", {})),
    )
    validate_env_dataset(dataset)
    return dataset
