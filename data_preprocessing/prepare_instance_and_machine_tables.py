import pandas as pd

try:
    from .pipeline_config import (
        INSTANCE_COLUMNS,
        INSTANCE_SUBSET_CSV,
        MACHINE_COLUMNS,
        MACHINE_META_CSV,
        PREPARED_INSTANCE_CSV,
        PREPARED_MACHINE_CSV,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        INSTANCE_COLUMNS,
        INSTANCE_SUBSET_CSV,
        MACHINE_COLUMNS,
        MACHINE_META_CSV,
        PREPARED_INSTANCE_CSV,
        PREPARED_MACHINE_CSV,
        ensure_directories,
    )


def load_raw_csv(path, expected_columns):
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    first_row = [value.strip() for value in first_line.split(",")]

    if first_row[: len(expected_columns)] == expected_columns:
        return pd.read_csv(path)

    return pd.read_csv(path, header=None, names=expected_columns)


def main() -> None:
    ensure_directories()

    if not INSTANCE_SUBSET_CSV.exists():
        raise FileNotFoundError(
            f"Missing subset file: {INSTANCE_SUBSET_CSV}. Run 02_create_instance_subset.py first."
        )

    if not MACHINE_META_CSV.exists():
        raise FileNotFoundError(
            f"Missing machine file: {MACHINE_META_CSV}. Run 03_extract_machine_metadata.py first."
        )

    instance = load_raw_csv(INSTANCE_SUBSET_CSV, INSTANCE_COLUMNS)
    machine = load_raw_csv(MACHINE_META_CSV, MACHINE_COLUMNS)

    instance = instance[
        [
            "instance_id",
            "machine_id",
            "cpu_util",
            "mem_util",
            "disk_io_util",
            "start_time",
            "end_time",
        ]
    ].rename(
        columns={
            "cpu_util": "cpu_task",
            "mem_util": "mem_task",
            "disk_io_util": "disk_task",
        }
    )

    machine = machine[["machine_id", "cpu", "mem", "disk"]].rename(
        columns={
            "cpu": "cpu_machine",
            "mem": "mem_machine",
            "disk": "disk_machine",
        }
    )

    instance = instance.dropna()
    machine = machine.dropna()

    numeric_instance_cols = [
        "cpu_task",
        "mem_task",
        "disk_task",
        "start_time",
        "end_time",
    ]
    numeric_machine_cols = ["cpu_machine", "mem_machine", "disk_machine"]

    for column in numeric_instance_cols:
        instance[column] = pd.to_numeric(instance[column], errors="coerce")

    for column in numeric_machine_cols:
        machine[column] = pd.to_numeric(machine[column], errors="coerce")

    instance = instance.dropna()
    machine = machine.dropna()

    instance["duration"] = instance["end_time"] - instance["start_time"]
    instance = instance[instance["duration"] > 0]

    instance.to_csv(PREPARED_INSTANCE_CSV, index=False)
    machine.to_csv(PREPARED_MACHINE_CSV, index=False)

    print(f"Saved prepared instance table: {PREPARED_INSTANCE_CSV}")
    print(f"Saved prepared machine table: {PREPARED_MACHINE_CSV}")
    print(f"Prepared instance shape: {instance.shape}")
    print(f"Prepared machine shape: {machine.shape}")


if __name__ == "__main__":
    main()
