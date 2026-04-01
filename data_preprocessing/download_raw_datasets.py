from urllib.request import urlretrieve

try:
    from .pipeline_config import (
        BATCH_INSTANCE_ARCHIVE,
        BATCH_INSTANCE_URL,
        MACHINE_META_ARCHIVE,
        MACHINE_META_URL,
        ensure_directories,
    )
except ImportError:
    from pipeline_config import (
        BATCH_INSTANCE_ARCHIVE,
        BATCH_INSTANCE_URL,
        MACHINE_META_ARCHIVE,
        MACHINE_META_URL,
        ensure_directories,
    )


def download_file(url: str, destination) -> None:
    if destination.exists():
        print(f"Skipping existing file: {destination}")
        return

    print(f"Downloading {url} -> {destination}")
    urlretrieve(url, destination)
    print(f"Saved: {destination}")


def main() -> None:
    ensure_directories()
    download_file(BATCH_INSTANCE_URL, BATCH_INSTANCE_ARCHIVE)
    download_file(MACHINE_META_URL, MACHINE_META_ARCHIVE)


if __name__ == "__main__":
    main()
