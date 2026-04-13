from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from rl_pipeline.env_dataset import RLEnvDataset


@dataclass(frozen=True)
class EpisodeSplits:
    train_episode_ids: list[int]
    val_episode_ids: list[int]
    test_episode_ids: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train_episode_ids": list(self.train_episode_ids),
            "val_episode_ids": list(self.val_episode_ids),
            "test_episode_ids": list(self.test_episode_ids),
        }


def _allocate_split_counts(
    *,
    total_count: int,
    fractions: tuple[float, float, float],
) -> tuple[int, int, int]:
    raw = np.asarray(fractions, dtype=np.float64) * float(total_count)
    counts = np.floor(raw).astype(int)

    remainder = int(total_count - int(counts.sum()))
    fractional_part = raw - counts
    allocation_order = np.argsort(-fractional_part)
    for index in allocation_order[:remainder]:
        counts[index] += 1

    while np.any(counts == 0):
        zero_indices = np.where(counts == 0)[0]
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise ValueError(
                "Unable to allocate at least one episode per split. "
                f"Received total episodes={total_count}, fractions={fractions}."
            )
        counts[donor] -= 1
        counts[int(zero_indices[0])] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def split_episode_ids_train_val_test(
    dataset: RLEnvDataset,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> EpisodeSplits:
    episode_ids = sorted(dataset.episodes["episode_id"].astype(int).unique().tolist())
    if len(episode_ids) < 3:
        raise ValueError(
            "Need at least 3 episodes to create train/val/test splits. "
            f"Received {len(episode_ids)} episode(s)."
        )

    fraction_sum = float(train_fraction + val_fraction + test_fraction)
    if not np.isclose(fraction_sum, 1.0, atol=1e-9):
        raise ValueError(
            "train_fraction + val_fraction + test_fraction must equal 1.0. "
            f"Received {fraction_sum:.8f}."
        )

    if min(float(train_fraction), float(val_fraction), float(test_fraction)) <= 0.0:
        raise ValueError(
            "All split fractions must be positive. "
            f"Received train={train_fraction}, val={val_fraction}, test={test_fraction}."
        )

    shuffled = np.asarray(episode_ids, dtype=np.int64).copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)

    train_count, val_count, test_count = _allocate_split_counts(
        total_count=len(shuffled),
        fractions=(float(train_fraction), float(val_fraction), float(test_fraction)),
    )

    train_ids = sorted(shuffled[:train_count].tolist())
    val_start = train_count
    val_end = train_count + val_count
    val_ids = sorted(shuffled[val_start:val_end].tolist())
    test_ids = sorted(shuffled[val_end : val_end + test_count].tolist())

    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            "Split allocation produced an empty split, which is not allowed. "
            f"Counts: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}."
        )

    return EpisodeSplits(
        train_episode_ids=train_ids,
        val_episode_ids=val_ids,
        test_episode_ids=test_ids,
    )


def save_episode_splits(
    splits: EpisodeSplits,
    output_path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(output_path)
    payload: dict[str, object] = {}
    if metadata:
        payload.update(metadata)
    payload.update(splits.as_dict())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_episode_splits(path: str | Path) -> EpisodeSplits:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EpisodeSplits(
        train_episode_ids=sorted({int(value) for value in payload["train_episode_ids"]}),
        val_episode_ids=sorted({int(value) for value in payload["val_episode_ids"]}),
        test_episode_ids=sorted({int(value) for value in payload["test_episode_ids"]}),
    )

