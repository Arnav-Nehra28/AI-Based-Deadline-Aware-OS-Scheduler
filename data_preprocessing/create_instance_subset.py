import tarfile

try:
    from .pipeline_config import (
        BATCH_INSTANCE_ARCHIVE,
        INSTANCE_SUBSET_CSV,
        INSTANCE_SUBSET_ROWS,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        BATCH_INSTANCE_ARCHIVE,
        INSTANCE_SUBSET_CSV,
        INSTANCE_SUBSET_ROWS,
        ensure_directories,
    )


def main() -> None:
    ensure_directories()

    if not BATCH_INSTANCE_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing archive: {BATCH_INSTANCE_ARCHIVE}. Run 01_download_raw_datasets.py first."
        )

    print(f"Creating subset from: {BATCH_INSTANCE_ARCHIVE}")

    with tarfile.open(BATCH_INSTANCE_ARCHIVE, "r:gz") as archive:
        member = archive.getmember("batch_instance.csv")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("Could not read batch_instance.csv from archive.")

        with INSTANCE_SUBSET_CSV.open("wb") as output_file:
            for line_number, line in enumerate(extracted):
                output_file.write(line)
                if line_number + 1 >= INSTANCE_SUBSET_ROWS:
                    break

    print(f"Saved subset to: {INSTANCE_SUBSET_CSV}")


if __name__ == "__main__":
    main()
