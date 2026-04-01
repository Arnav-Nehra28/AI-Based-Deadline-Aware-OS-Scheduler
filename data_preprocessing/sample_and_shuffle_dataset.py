import pandas as pd

try:
    from .pipeline_config import (
        FINAL_DATASET_CSV,
        FINAL_SAMPLE_SIZE,
        MODEL_READY_DATASET_CSV,
        RANDOM_STATE,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        FINAL_DATASET_CSV,
        FINAL_SAMPLE_SIZE,
        MODEL_READY_DATASET_CSV,
        RANDOM_STATE,
        ensure_directories,
    )


def main() -> None:
    ensure_directories()

    if not MODEL_READY_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing model-ready dataset: {MODEL_READY_DATASET_CSV}. "
            "Run build_model_ready_dataset.py first."
        )

    dataset = pd.read_csv(MODEL_READY_DATASET_CSV)

    sample_size = min(FINAL_SAMPLE_SIZE, len(dataset))
    dataset = dataset.sample(n=sample_size, random_state=RANDOM_STATE)
    dataset = dataset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    dataset.to_csv(FINAL_DATASET_CSV, index=False)

    print(f"Saved sampled and shuffled dataset: {FINAL_DATASET_CSV}")
    print(f"Final dataset shape: {dataset.shape}")


if __name__ == "__main__":
    main()
