import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from .pipeline_config import MERGED_DATASET_CSV, SCALED_DATASET_CSV, ensure_directories
except ImportError:
    from pipeline_config import MERGED_DATASET_CSV, SCALED_DATASET_CSV, ensure_directories

RESOURCE_COLUMNS = [
    "cpu_task",
    "mem_task",
    "disk_task",
    "cpu_machine",
    "mem_machine",
    "disk_machine",
]


def main() -> None:
    ensure_directories()

    if not MERGED_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing merged dataset: {MERGED_DATASET_CSV}. Run 05_merge_task_and_machine_data.py first."
        )

    dataset = pd.read_csv(MERGED_DATASET_CSV)
    scaler = MinMaxScaler()
    dataset[RESOURCE_COLUMNS] = scaler.fit_transform(dataset[RESOURCE_COLUMNS])

    dataset.to_csv(SCALED_DATASET_CSV, index=False)

    print(f"Saved scaled dataset: {SCALED_DATASET_CSV}")
    print(f"Scaled dataset shape: {dataset.shape}")


if __name__ == "__main__":
    main()
