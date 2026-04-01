from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from rl_pipeline.env_dataset import RLEnvDataset, save_env_dataset

try:
    from .pipeline_config import (
        BATCH_INSTANCE_ARCHIVE,
        MERGED_DATASET_CSV,
        MODEL_READY_DATASET_CSV,
        PREPARED_INSTANCE_CSV,
        PREPARED_MACHINE_CSV,
        PROCESSED_SPLIT_PATHS,
        RANDOM_STATE,
        RL_ENV_DATASET_JSON_GZ,
        RL_ENV_EPISODE_LENGTH,
        RL_ENV_MIN_VALID_DECISIONS,
        RL_ENV_RAW_TASK_SCAN_LIMIT,
        RL_ENV_TARGET_EPISODES,
        SCALED_DATASET_CSV,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        BATCH_INSTANCE_ARCHIVE,
        MERGED_DATASET_CSV,
        MODEL_READY_DATASET_CSV,
        PREPARED_INSTANCE_CSV,
        PREPARED_MACHINE_CSV,
        PROCESSED_SPLIT_PATHS,
        RANDOM_STATE,
        RL_ENV_DATASET_JSON_GZ,
        RL_ENV_EPISODE_LENGTH,
        RL_ENV_MIN_VALID_DECISIONS,
        RL_ENV_RAW_TASK_SCAN_LIMIT,
        RL_ENV_TARGET_EPISODES,
        SCALED_DATASET_CSV,
        ensure_directories,
    )


RAW_INSTANCE_COLUMNS = [
    "task_id",
    "arrival_time",
    "duration",
    "cpu_demand",
    "mem_demand",
    "disk_demand",
    "historical_machine_id",
]


def _sanitize_machine_catalog(machine_frame: pd.DataFrame) -> pd.DataFrame:
    machines = machine_frame.copy()
    machines["machine_id"] = machines["machine_id"].astype(str)
    for column in ["cpu_capacity", "mem_capacity", "disk_capacity"]:
        machines[column] = pd.to_numeric(machines[column], errors="coerce").clip(lower=0)

    machines = machines.dropna(subset=["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"])
    machines = (
        machines.groupby("machine_id", as_index=False)[["cpu_capacity", "mem_capacity", "disk_capacity"]]
        .max()
        .sort_values("machine_id")
        .reset_index(drop=True)
    )
    return machines[(machines[["cpu_capacity", "mem_capacity", "disk_capacity"]].sum(axis=1) > 0)].reset_index(
        drop=True
    )


def _prepare_task_table(task_frame: pd.DataFrame) -> pd.DataFrame:
    tasks = task_frame.copy()
    tasks["task_id"] = tasks["task_id"].astype(str)
    tasks["historical_machine_id"] = tasks["historical_machine_id"].astype(str)

    numeric_columns = ["arrival_time", "duration", "cpu_demand", "mem_demand", "disk_demand"]
    for column in numeric_columns:
        tasks[column] = pd.to_numeric(tasks[column], errors="coerce")

    tasks = tasks.dropna(subset=["task_id", "historical_machine_id", *numeric_columns])
    tasks = tasks[tasks["duration"] > 0].copy()
    for column in ["cpu_demand", "mem_demand", "disk_demand"]:
        tasks[column] = tasks[column].clip(lower=0)

    tasks = tasks.sort_values(["arrival_time", "task_id"]).drop_duplicates("task_id").reset_index(drop=True)
    return tasks


def _estimate_machine_capacities_from_tasks(
    tasks: pd.DataFrame,
    safety_margin: float = 1.10,
    minimum_capacity: float = 1.0,
) -> pd.DataFrame:
    event_rows: list[tuple[str, float, int, float, float, float]] = []
    for row in tasks.itertuples(index=False):
        event_rows.append(
            (
                str(row.historical_machine_id),
                float(row.arrival_time),
                1,
                float(row.cpu_demand),
                float(row.mem_demand),
                float(row.disk_demand),
            )
        )
        event_rows.append(
            (
                str(row.historical_machine_id),
                float(row.arrival_time + row.duration),
                -1,
                float(row.cpu_demand),
                float(row.mem_demand),
                float(row.disk_demand),
            )
        )

    events = pd.DataFrame(
        event_rows,
        columns=["machine_id", "event_time", "direction", "cpu_demand", "mem_demand", "disk_demand"],
    ).sort_values(["machine_id", "event_time", "direction"])

    machine_rows: list[dict[str, float | str]] = []
    for machine_id, frame in events.groupby("machine_id", sort=True):
        cpu_in_use = 0.0
        mem_in_use = 0.0
        disk_in_use = 0.0
        peak_cpu = 0.0
        peak_mem = 0.0
        peak_disk = 0.0

        for event in frame.itertuples(index=False):
            cpu_in_use += float(event.direction) * float(event.cpu_demand)
            mem_in_use += float(event.direction) * float(event.mem_demand)
            disk_in_use += float(event.direction) * float(event.disk_demand)
            peak_cpu = max(peak_cpu, cpu_in_use)
            peak_mem = max(peak_mem, mem_in_use)
            peak_disk = max(peak_disk, disk_in_use)

        machine_rows.append(
            {
                "machine_id": machine_id,
                "cpu_capacity": max(minimum_capacity, peak_cpu * safety_margin),
                "mem_capacity": max(minimum_capacity, peak_mem * safety_margin),
                "disk_capacity": max(minimum_capacity, peak_disk * safety_margin),
            }
        )

    return pd.DataFrame(machine_rows)


def _stream_raw_batch_tasks(max_tasks: int, max_scan_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if not BATCH_INSTANCE_ARCHIVE.exists():
        raise FileNotFoundError(f"Missing raw trace archive: {BATCH_INSTANCE_ARCHIVE}")

    task_rows: list[dict[str, object]] = []
    scanned_rows = 0

    with tarfile.open(BATCH_INSTANCE_ARCHIVE, "r|gz") as archive:
        member = archive.next()
        if member is None:
            raise ValueError(f"No file entries found in archive: {BATCH_INSTANCE_ARCHIVE}")
        file_obj = archive.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Could not stream archive member: {member.name}")

        while len(task_rows) < max_tasks and scanned_rows < max_scan_rows:
            raw_line = file_obj.readline()
            if not raw_line:
                break
            scanned_rows += 1

            parts = raw_line.decode("utf-8", errors="ignore").strip().split(",")
            if len(parts) < 14:
                continue

            instance_id = parts[0].strip()
            status = parts[4].strip()
            start_time = parts[5].strip()
            end_time = parts[6].strip()
            machine_id = parts[7].strip()
            cpu_util = parts[10].strip()
            mem_util = parts[11].strip()
            disk_io_util = parts[12].strip()

            if not instance_id or not machine_id or status != "Terminated":
                continue

            try:
                arrival_time = float(start_time)
                finish_time = float(end_time)
                duration = finish_time - arrival_time
                cpu_demand = float(cpu_util)
                mem_demand = float(mem_util)
                disk_demand = float(disk_io_util)
            except ValueError:
                continue

            if duration <= 0:
                continue

            task_rows.append(
                {
                    "task_id": instance_id,
                    "arrival_time": arrival_time,
                    "duration": duration,
                    "cpu_demand": cpu_demand,
                    "mem_demand": mem_demand,
                    "disk_demand": disk_demand,
                    "historical_machine_id": machine_id,
                }
            )

    tasks = _prepare_task_table(pd.DataFrame(task_rows, columns=RAW_INSTANCE_COLUMNS))
    if tasks.empty:
        raise ValueError("No valid tasks could be streamed from the raw trace archive.")

    machines = _estimate_machine_capacities_from_tasks(tasks)
    return tasks, machines, {"source_kind": "raw_batch_instance_archive"}


def _load_source_tables(
    required_task_count: int,
    raw_task_scan_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if BATCH_INSTANCE_ARCHIVE.exists():
        try:
            return _stream_raw_batch_tasks(max_tasks=required_task_count, max_scan_rows=raw_task_scan_limit)
        except (FileNotFoundError, ValueError, tarfile.TarError):
            pass

    if PREPARED_INSTANCE_CSV.exists() and PREPARED_MACHINE_CSV.exists():
        instance = pd.read_csv(PREPARED_INSTANCE_CSV)
        machine = pd.read_csv(PREPARED_MACHINE_CSV)
        tasks = instance.rename(
            columns={
                "instance_id": "task_id",
                "start_time": "arrival_time",
                "machine_id": "historical_machine_id",
                "cpu_task": "cpu_demand",
                "mem_task": "mem_demand",
                "disk_task": "disk_demand",
            }
        )[
            ["task_id", "arrival_time", "duration", "cpu_demand", "mem_demand", "disk_demand", "historical_machine_id"]
        ]
        machines = machine.rename(
            columns={
                "cpu_machine": "cpu_capacity",
                "mem_machine": "mem_capacity",
                "disk_machine": "disk_capacity",
            }
        )[["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]]
        return tasks, machines, {"source_kind": "prepared_tables"}

    if MERGED_DATASET_CSV.exists():
        merged = pd.read_csv(MERGED_DATASET_CSV)
        tasks = merged.rename(
            columns={
                "instance_id": "task_id",
                "start_time": "arrival_time",
                "machine_id": "historical_machine_id",
                "cpu_task": "cpu_demand",
                "mem_task": "mem_demand",
                "disk_task": "disk_demand",
            }
        )[
            ["task_id", "arrival_time", "duration", "cpu_demand", "mem_demand", "disk_demand", "historical_machine_id"]
        ]
        machines = merged.rename(
            columns={
                "cpu_machine": "cpu_capacity",
                "mem_machine": "mem_capacity",
                "disk_machine": "disk_capacity",
            }
        )[["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]]
        return tasks, machines, {"source_kind": "merged_dataset"}

    if SCALED_DATASET_CSV.exists():
        scaled = pd.read_csv(SCALED_DATASET_CSV)
        task_id_column = "instance_id" if "instance_id" in scaled.columns else None
        if task_id_column is None:
            scaled = scaled.reset_index(names="task_id")
        else:
            scaled = scaled.rename(columns={task_id_column: "task_id"})

        tasks = scaled.rename(
            columns={
                "start_time": "arrival_time",
                "machine_id": "historical_machine_id",
                "cpu_task": "cpu_demand",
                "mem_task": "mem_demand",
                "disk_task": "disk_demand",
            }
        )[
            ["task_id", "arrival_time", "duration", "cpu_demand", "mem_demand", "disk_demand", "historical_machine_id"]
        ]
        machines = scaled.rename(
            columns={
                "cpu_machine": "cpu_capacity",
                "mem_machine": "mem_capacity",
                "disk_machine": "disk_capacity",
            }
        )[["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]]
        return tasks, machines, {"source_kind": "scaled_dataset"}

    if MODEL_READY_DATASET_CSV.exists():
        model_ready = pd.read_csv(MODEL_READY_DATASET_CSV).reset_index(names="task_id")
        tasks = model_ready.rename(
            columns={
                "start_time": "arrival_time",
                "machine_id": "historical_machine_id",
                "cpu_task": "cpu_demand",
                "mem_task": "mem_demand",
                "disk_task": "disk_demand",
            }
        )[
            ["task_id", "arrival_time", "duration", "cpu_demand", "mem_demand", "disk_demand", "historical_machine_id"]
        ]
        machines = model_ready.rename(
            columns={
                "cpu_machine": "cpu_capacity",
                "mem_machine": "mem_capacity",
                "disk_machine": "disk_capacity",
            }
        )[["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]]
        return tasks, machines, {"source_kind": "model_ready_dataset"}

    split_frames = []
    for split_name, path in PROCESSED_SPLIT_PATHS.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_split"] = split_name
            split_frames.append(frame)

    if split_frames:
        processed = pd.concat(split_frames, ignore_index=True).reset_index(names="task_id")
        tasks = processed.rename(
            columns={
                "start_time": "arrival_time",
                "machine_id": "historical_machine_id",
                "cpu_task": "cpu_demand",
                "mem_task": "mem_demand",
                "disk_task": "disk_demand",
            }
        )[
            ["task_id", "arrival_time", "duration", "cpu_demand", "mem_demand", "disk_demand", "historical_machine_id"]
        ]
        machines = processed.rename(
            columns={
                "cpu_machine": "cpu_capacity",
                "mem_machine": "mem_capacity",
                "disk_machine": "disk_capacity",
            }
        )[["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]]
        return tasks, machines, {"source_kind": "processed_splits"}

    raise FileNotFoundError(
        "No source tables available for RL env dataset construction. "
        "Run preprocessing first or provide the raw batch instance archive."
    )


def _ensure_dataset_scale(
    tasks: pd.DataFrame,
    target_episode_count: int,
    episode_length: int,
    random_state: int,
) -> pd.DataFrame:
    required_tasks = target_episode_count * episode_length
    if len(tasks) >= required_tasks:
        return tasks.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    base_tasks = tasks.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)
    duration_scale = max(float(base_tasks["duration"].median()), 1.0)
    episode_span = max(float(base_tasks["arrival_time"].max() - base_tasks["arrival_time"].min()), duration_scale)

    synthetic_batches = [base_tasks]
    next_task_offset = 0
    while sum(len(batch) for batch in synthetic_batches) < required_tasks:
        sampled = base_tasks.sample(n=min(len(base_tasks), required_tasks), replace=True, random_state=int(rng.integers(1_000_000)))
        sampled = sampled.reset_index(drop=True).copy()
        replication_index = len(synthetic_batches)
        sampled["task_id"] = sampled["task_id"].astype(str) + f"_rep{replication_index}_" + sampled.index.astype(str)
        sampled["arrival_time"] = (
            sampled["arrival_time"]
            - float(sampled["arrival_time"].min())
            + replication_index * (episode_span + duration_scale)
        )
        sampled["duration"] = np.maximum(
            1.0,
            sampled["duration"] * rng.uniform(0.9, 1.1, size=len(sampled)),
        )
        sampled["cpu_demand"] = np.maximum(0.0, sampled["cpu_demand"] * rng.uniform(0.95, 1.05, size=len(sampled)))
        sampled["mem_demand"] = np.maximum(0.0, sampled["mem_demand"] * rng.uniform(0.95, 1.05, size=len(sampled)))
        sampled["disk_demand"] = np.maximum(
            0.0,
            sampled["disk_demand"] * rng.uniform(0.95, 1.05, size=len(sampled)),
        )
        synthetic_batches.append(sampled)
        next_task_offset += len(sampled)

    expanded = pd.concat(synthetic_batches, ignore_index=True)
    return expanded.sort_values(["arrival_time", "task_id"]).head(required_tasks).reset_index(drop=True)


def build_rl_env_dataset(
    output_path: Path = RL_ENV_DATASET_JSON_GZ,
    episode_length: int = RL_ENV_EPISODE_LENGTH,
    min_valid_decisions: int = RL_ENV_MIN_VALID_DECISIONS,
    max_episodes: int | None = RL_ENV_TARGET_EPISODES,
    random_state: int = RANDOM_STATE,
    raw_task_scan_limit: int = RL_ENV_RAW_TASK_SCAN_LIMIT,
) -> RLEnvDataset:
    ensure_directories()

    target_episode_count = RL_ENV_TARGET_EPISODES if max_episodes is None else max_episodes
    required_task_count = max(target_episode_count * episode_length * 2, raw_task_scan_limit)

    tasks, machines, source_metadata = _load_source_tables(
        required_task_count=required_task_count,
        raw_task_scan_limit=max(raw_task_scan_limit, required_task_count),
    )
    tasks = _prepare_task_table(tasks)
    machines = _sanitize_machine_catalog(machines)

    tasks = tasks[tasks["historical_machine_id"].isin(set(machines["machine_id"].tolist()))].reset_index(drop=True)
    if tasks.empty:
        raise ValueError("No valid tasks remain after aligning tasks with the machine catalog.")

    tasks = _ensure_dataset_scale(
        tasks=tasks,
        target_episode_count=target_episode_count,
        episode_length=episode_length,
        random_state=random_state,
    )
    if source_metadata["source_kind"] != "raw_batch_instance_archive" and len(tasks) < target_episode_count * episode_length:
        machines = _estimate_machine_capacities_from_tasks(tasks)

    candidate_windows: list[pd.DataFrame] = []
    max_possible_episodes = len(tasks) // episode_length
    actual_episode_count = min(target_episode_count, max_possible_episodes)
    for episode_id in range(actual_episode_count):
        start = episode_id * episode_length
        end = start + episode_length
        window = tasks.iloc[start:end].copy()
        if len(window) < min_valid_decisions:
            continue
        candidate_windows.append(window)

    if len(candidate_windows) < target_episode_count:
        raise ValueError(
            f"Only built {len(candidate_windows)} RL episodes; required at least {target_episode_count}. "
            "Increase raw_task_scan_limit or provide richer preprocessing inputs."
        )

    task_rows: list[pd.DataFrame] = []
    episode_rows: list[dict[str, object]] = []
    for episode_id, window in enumerate(candidate_windows):
        window = window.sort_values(["arrival_time", "task_id"]).reset_index(drop=True)
        window["episode_id"] = episode_id
        window["task_index"] = np.arange(len(window), dtype=int)
        task_rows.append(window)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "task_count": int(len(window)),
                "start_time": float(window["arrival_time"].min()),
                "end_time": float((window["arrival_time"] + window["duration"]).max()),
                "source_kind": source_metadata["source_kind"],
            }
        )

    dataset = RLEnvDataset(
        tasks=pd.concat(task_rows, ignore_index=True)[
            [
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
        ],
        machines=machines[["machine_id", "cpu_capacity", "mem_capacity", "disk_capacity"]],
        episodes=pd.DataFrame(episode_rows),
        metadata={
            "episode_length": int(episode_length),
            "min_valid_decisions": int(min_valid_decisions),
            "target_episode_count": int(target_episode_count),
            "random_state": int(random_state),
            "source_kind": source_metadata["source_kind"],
        },
    )
    save_env_dataset(dataset, output_path)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an env-ready dataset artifact for the RL scheduler.")
    parser.add_argument("--output", type=Path, default=RL_ENV_DATASET_JSON_GZ)
    parser.add_argument("--episode-length", type=int, default=RL_ENV_EPISODE_LENGTH)
    parser.add_argument("--min-valid-decisions", type=int, default=RL_ENV_MIN_VALID_DECISIONS)
    parser.add_argument("--max-episodes", type=int, default=RL_ENV_TARGET_EPISODES)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--raw-task-scan-limit", type=int, default=RL_ENV_RAW_TASK_SCAN_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_rl_env_dataset(
        output_path=args.output,
        episode_length=args.episode_length,
        min_valid_decisions=args.min_valid_decisions,
        max_episodes=args.max_episodes,
        random_state=args.random_state,
        raw_task_scan_limit=args.raw_task_scan_limit,
    )

    print(f"Saved RL env dataset: {args.output}")
    print(f"Tasks: {dataset.tasks.shape}")
    print(f"Machines: {dataset.machines.shape}")
    print(f"Episodes: {dataset.episodes.shape}")
    print(f"Source kind: {dataset.metadata['source_kind']}")


if __name__ == "__main__":
    main()
