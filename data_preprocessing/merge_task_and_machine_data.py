import pandas as pd

try:
    from .pipeline_config import (
        MERGED_DATASET_CSV,
        PREPARED_INSTANCE_CSV,
        PREPARED_MACHINE_CSV,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        MERGED_DATASET_CSV,
        PREPARED_INSTANCE_CSV,
        PREPARED_MACHINE_CSV,
        ensure_directories,
    )


def main() -> None:
    ensure_directories()

    if not PREPARED_INSTANCE_CSV.exists():
        raise FileNotFoundError(
            f"Missing prepared instance file: {PREPARED_INSTANCE_CSV}. "
            "Run 04_prepare_instance_and_machine_tables.py first."
        )

    if not PREPARED_MACHINE_CSV.exists():
        raise FileNotFoundError(
            f"Missing prepared machine file: {PREPARED_MACHINE_CSV}. "
            "Run 04_prepare_instance_and_machine_tables.py first."
        )

    instance = pd.read_csv(PREPARED_INSTANCE_CSV)
    machine = pd.read_csv(PREPARED_MACHINE_CSV)

    dataset = pd.merge(instance, machine, on="machine_id", how="inner")
    dataset.to_csv(MERGED_DATASET_CSV, index=False)

    print(f"Saved merged dataset: {MERGED_DATASET_CSV}")
    print(f"Merged dataset shape: {dataset.shape}")


if __name__ == "__main__":
    main()
