from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

BATCH_INSTANCE_URL = (
    "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_instance.tar.gz"
)
MACHINE_META_URL = (
    "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_meta.tar.gz"
)

BATCH_INSTANCE_ARCHIVE = RAW_DIR / "batch_instance.tar.gz"
MACHINE_META_ARCHIVE = RAW_DIR / "machine_meta.tar.gz"
BATCH_INSTANCE_CSV = RAW_DIR / "batch_instance.csv"
MACHINE_META_CSV = RAW_DIR / "machine_meta.csv"

INSTANCE_SUBSET_CSV = INTERIM_DIR / "instance_subset.csv"
PREPARED_INSTANCE_CSV = INTERIM_DIR / "prepared_instance.csv"
PREPARED_MACHINE_CSV = INTERIM_DIR / "prepared_machine.csv"
MERGED_DATASET_CSV = INTERIM_DIR / "merged_dataset.csv"
SCALED_DATASET_CSV = INTERIM_DIR / "scaled_dataset.csv"
MODEL_READY_DATASET_CSV = INTERIM_DIR / "model_ready_dataset.csv"
FINAL_DATASET_CSV = INTERIM_DIR / "sampled_shuffled_dataset.csv"
RL_ENV_DATASET_JSON_GZ = INTERIM_DIR / "rl_env_dataset.json.gz"

TRAIN_DATASET_CSV = PROCESSED_DIR / "processed_train.csv"
VAL_DATASET_CSV = PROCESSED_DIR / "processed_val.csv"
TEST_DATASET_CSV = PROCESSED_DIR / "processed_test.csv"

VALIDATION_REPORT_JSON = REPORTS_DIR / "processed_data_validation.json"
PROFILE_REPORT_JSON = REPORTS_DIR / "processed_data_profile.json"
PROFILE_REPORT_MD = REPORTS_DIR / "processed_data_profile.md"
RL_ENV_VALIDATION_REPORT_JSON = REPORTS_DIR / "rl_env_validation.json"

PROCESSED_SPLIT_PATHS = {
    "train": TRAIN_DATASET_CSV,
    "validation": VAL_DATASET_CSV,
    "test": TEST_DATASET_CSV,
}

EXPECTED_PROCESSED_COLUMNS = [
    "machine_id",
    "cpu_task",
    "mem_task",
    "disk_task",
    "start_time",
    "duration",
    "cpu_machine",
    "mem_machine",
    "disk_machine",
    "cpu_ratio",
    "mem_ratio",
    "disk_ratio",
    "cpu_gap",
    "mem_gap",
    "resource_pressure",
    "task_hour",
]

SCALED_FEATURE_COLUMNS = [
    "cpu_task",
    "mem_task",
    "disk_task",
    "cpu_machine",
    "mem_machine",
    "disk_machine",
]

NUMERIC_PROCESSED_COLUMNS = [
    "cpu_task",
    "mem_task",
    "disk_task",
    "start_time",
    "duration",
    "cpu_machine",
    "mem_machine",
    "disk_machine",
    "cpu_ratio",
    "mem_ratio",
    "disk_ratio",
    "cpu_gap",
    "mem_gap",
    "resource_pressure",
    "task_hour",
]

RATIO_COLUMNS = ["cpu_ratio", "mem_ratio", "disk_ratio"]
GAP_COLUMNS = ["cpu_gap", "mem_gap"]
DERIVED_FEATURE_COLUMNS = [
    "cpu_ratio",
    "mem_ratio",
    "disk_ratio",
    "cpu_gap",
    "mem_gap",
    "resource_pressure",
    "task_hour",
]

INSTANCE_COLUMNS = [
    "instance_id",
    "task_id",
    "job_id",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "machine_id",
    "plan_cpu",
    "plan_mem",
    "cpu_util",
    "mem_util",
    "disk_io_util",
    "network_util",
]

MACHINE_COLUMNS = [
    "machine_id",
    "timestamp",
    "cpu",
    "mem",
    "disk",
    "status",
    "cpu_count",
    "mem_size",
    "disk_size",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "p9",
    "p10",
    "p11",
    "p12",
]

INSTANCE_SUBSET_ROWS = 500000
FINAL_SAMPLE_SIZE = 50000
RL_ENV_EPISODE_LENGTH = 128
RL_ENV_MIN_VALID_DECISIONS = 128
RL_ENV_TARGET_EPISODES = 100
RL_ENV_RAW_TASK_SCAN_LIMIT = 250000
RANDOM_STATE = 42


def ensure_directories() -> None:
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
