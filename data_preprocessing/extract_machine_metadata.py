import tarfile

try:
    from .pipeline_config import MACHINE_META_ARCHIVE, MACHINE_META_CSV, ensure_directories
except ImportError:
    from pipeline_config import MACHINE_META_ARCHIVE, MACHINE_META_CSV, ensure_directories


def main() -> None:
    ensure_directories()

    if not MACHINE_META_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing archive: {MACHINE_META_ARCHIVE}. Run 01_download_raw_datasets.py first."
        )

    if MACHINE_META_CSV.exists():
        print(f"Skipping existing file: {MACHINE_META_CSV}")
        return

    print(f"Extracting machine metadata from: {MACHINE_META_ARCHIVE}")

    with tarfile.open(MACHINE_META_ARCHIVE, "r:gz") as archive:
        member = archive.getmember("machine_meta.csv")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("Could not read machine_meta.csv from archive.")

        with MACHINE_META_CSV.open("wb") as output_file:
            output_file.write(extracted.read())

    print(f"Saved machine metadata to: {MACHINE_META_CSV}")


if __name__ == "__main__":
    main()
