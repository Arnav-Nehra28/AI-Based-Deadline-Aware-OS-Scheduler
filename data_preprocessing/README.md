# Data Preprocessing

This folder contains the dataset preprocessing workflow split out from the old `Minor_Project.ipynb` notebook.

The original notebook has been preserved here as:

- `legacy_minor_project_preprocessing.ipynb`

Use the scripts below in order:

1. `download_raw_datasets.py`
2. `create_instance_subset.py`
3. `extract_machine_metadata.py`
4. `prepare_instance_and_machine_tables.py`
5. `merge_task_and_machine_data.py`
6. `scale_resource_features.py`
7. `build_model_ready_dataset.py`
8. `sample_and_shuffle_dataset.py`
9. `split_processed_dataset.py`
10. `validate_processed_data.py`
11. `profile_processed_data.py`
12. `build_rl_env_dataset.py`

You can also run the full workflow with:

```bash
python -m data_preprocessing
```

Directory layout used by this workflow:

- `data/raw/` stores downloaded archives and extracted raw CSVs
- `data/interim/` stores intermediate preprocessing outputs
- `data/processed/` stores the final train, validation, and test CSV files
- `data/reports/` stores validation and profiling outputs

## Dataset Contract

The processed dataset is expected to contain these columns:

| Column | Type | Description |
| --- | --- | --- |
| `machine_id` | categorical/string | Target machine assigned to the task |
| `cpu_task` | numeric | Scaled task CPU utilization |
| `mem_task` | numeric | Scaled task memory utilization |
| `disk_task` | numeric | Scaled task disk utilization |
| `start_time` | numeric/integer | Task start timestamp from the trace |
| `duration` | numeric/integer | Positive runtime computed as `end_time - start_time` |
| `cpu_machine` | numeric | Scaled machine CPU capacity |
| `mem_machine` | numeric | Scaled machine memory capacity |
| `disk_machine` | numeric | Scaled machine disk capacity |
| `cpu_ratio` | numeric | `cpu_task / cpu_machine` style load ratio |
| `mem_ratio` | numeric | `mem_task / mem_machine` style load ratio |
| `disk_ratio` | numeric | `disk_task / disk_machine` style load ratio |
| `cpu_gap` | numeric | `cpu_machine - cpu_task` |
| `mem_gap` | numeric | `mem_machine - mem_task` |
| `resource_pressure` | numeric | Combined resource pressure indicator |
| `task_hour` | numeric/integer | Hour-of-day extracted from `start_time` |

Current output files:

- `data/processed/processed_train.csv`
- `data/processed/processed_val.csv`
- `data/processed/processed_test.csv`

Validation expectations:

- All expected columns must be present.
- No null or non-finite values should exist.
- `duration` must be positive.
- `task_hour` must remain within `0-23`.
- Scaled resource columns should remain within `[0, 1]`.

Additional pipeline behavior:

- Duplicate rows are removed in `build_model_ready_dataset.py`.
- `split_processed_dataset.py` automatically moves unseen `machine_id` rows from validation/test into train, so every eval label is train-covered.

Generated reports:

- `data/reports/processed_data_validation.json`
- `data/reports/processed_data_profile.json`
- `data/reports/processed_data_profile.md`

RL environment artifact:

- `data/interim/rl_env_dataset.json.gz`

Use `build_rl_env_dataset.py` to build an environment-ready scheduling artifact with:

- a machine catalog
- time-ordered task arrivals
- episode boundaries for RL training
- historical assignments retained only as optional shaping metadata
- a default scale target of at least `100` episodes with `128` tasks each when raw trace data is available

The current processed files are already available in `data/processed/`.
