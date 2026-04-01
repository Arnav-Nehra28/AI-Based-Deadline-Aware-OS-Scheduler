import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .pipeline_config import (
        FINAL_DATASET_CSV,
        RANDOM_STATE,
        TEST_DATASET_CSV,
        TRAIN_DATASET_CSV,
        VAL_DATASET_CSV,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        FINAL_DATASET_CSV,
        RANDOM_STATE,
        TEST_DATASET_CSV,
        TRAIN_DATASET_CSV,
        VAL_DATASET_CSV,
        ensure_directories,
    )


def _move_unseen_machine_ids_to_train(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train_machine_ids = set(train_data["machine_id"])
    unseen_mask = ~eval_data["machine_id"].isin(train_machine_ids)

    moved_rows = eval_data.loc[unseen_mask]
    kept_eval_rows = eval_data.loc[~unseen_mask]

    if moved_rows.empty:
        return train_data, kept_eval_rows.reset_index(drop=True), 0

    updated_train = pd.concat([train_data, moved_rows], ignore_index=True)
    return updated_train, kept_eval_rows.reset_index(drop=True), int(len(moved_rows))


def main() -> None:
    ensure_directories()

    if not FINAL_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing final dataset: {FINAL_DATASET_CSV}. Run 07_sample_and_shuffle_dataset.py first."
        )

    dataset = pd.read_csv(FINAL_DATASET_CSV)

    if len(dataset) < 3:
        train_data = dataset.copy()
        val_data = dataset.iloc[0:0].copy()
        test_data = dataset.iloc[0:0].copy()
    else:
        train_data, temp_data = train_test_split(
            dataset,
            test_size=0.30,
            random_state=RANDOM_STATE,
        )

        if len(temp_data) < 2:
            val_data = temp_data.copy()
            test_data = temp_data.iloc[0:0].copy()
        else:
            val_data, test_data = train_test_split(
                temp_data,
                test_size=0.50,
                random_state=RANDOM_STATE,
            )

    train_data, val_data, moved_from_val = _move_unseen_machine_ids_to_train(
        train_data=train_data,
        eval_data=val_data,
    )
    train_data, test_data, moved_from_test = _move_unseen_machine_ids_to_train(
        train_data=train_data,
        eval_data=test_data,
    )

    train_data = train_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train_data.to_csv(TRAIN_DATASET_CSV, index=False)
    val_data.to_csv(VAL_DATASET_CSV, index=False)
    test_data.to_csv(TEST_DATASET_CSV, index=False)

    print(f"Train: {train_data.shape} -> {TRAIN_DATASET_CSV}")
    print(f"Validation: {val_data.shape} -> {VAL_DATASET_CSV}")
    print(f"Test: {test_data.shape} -> {TEST_DATASET_CSV}")
    print(f"Moved from validation to train (unseen machine_id): {moved_from_val}")
    print(f"Moved from test to train (unseen machine_id): {moved_from_test}")


if __name__ == "__main__":
    main()
