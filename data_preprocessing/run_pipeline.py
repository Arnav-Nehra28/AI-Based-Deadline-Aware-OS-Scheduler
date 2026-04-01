from .build_model_ready_dataset import main as build_model_ready_dataset
from .create_instance_subset import main as create_instance_subset
from .download_raw_datasets import main as download_raw_datasets
from .extract_machine_metadata import main as extract_machine_metadata
from .merge_task_and_machine_data import main as merge_task_and_machine_data
from .profile_processed_data import main as profile_processed_data
from .prepare_instance_and_machine_tables import main as prepare_instance_and_machine_tables
from .sample_and_shuffle_dataset import main as sample_and_shuffle_dataset
from .scale_resource_features import main as scale_resource_features
from .split_processed_dataset import main as split_processed_dataset
from .validate_processed_data import main as validate_processed_data


PIPELINE_STEPS = [
    ("Download raw datasets", download_raw_datasets),
    ("Create instance subset", create_instance_subset),
    ("Extract machine metadata", extract_machine_metadata),
    ("Prepare instance and machine tables", prepare_instance_and_machine_tables),
    ("Merge task and machine data", merge_task_and_machine_data),
    ("Scale resource features", scale_resource_features),
    ("Build model-ready dataset", build_model_ready_dataset),
    ("Sample and shuffle dataset", sample_and_shuffle_dataset),
    ("Split processed dataset", split_processed_dataset),
    ("Validate processed dataset", validate_processed_data),
    ("Profile processed dataset", profile_processed_data),
]


def main() -> None:
    for index, (label, step) in enumerate(PIPELINE_STEPS, start=1):
        print(f"\n[{index}/{len(PIPELINE_STEPS)}] {label}")
        step()

    print("\nPreprocessing pipeline completed.")


if __name__ == "__main__":
    main()
