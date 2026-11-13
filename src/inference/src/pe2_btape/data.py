from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: int
    text: str
    label: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Split:
    pool: list[Example]
    val: list[Example]
    test: list[Example]

    def manifest(self) -> dict[str, list[int]]:
        return {
            "pool_ids": [example.id for example in self.pool],
            "val_ids": [example.id for example in self.val],
            "test_ids": [example.id for example in self.test],
        }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("Dataset must contain text and label columns")
        for row_id, row in enumerate(reader):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                raise ValueError(f"Empty text or label at CSV data row {row_id + 2}")
            examples.append(Example(row_id, text, label))
    if not examples:
        raise ValueError("Dataset is empty")
    return examples


def three_way_split(examples: list[Example], seed: int) -> Split:
    """Perform the paper's single deterministic 30/30/40 random split."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_pool = round(len(shuffled) * 0.30)
    n_val = round(len(shuffled) * 0.30)
    return Split(
        pool=shuffled[:n_pool],
        val=shuffled[n_pool : n_pool + n_val],
        test=shuffled[n_pool + n_val :],
    )


def stratified_three_way_split(examples: list[Example], seed: int) -> Split:
    """Create one deterministic, label-stratified 30/30/40 split.

    Every seed independently partitions the complete dataset; these are
    repeated holdout splits, not cross-validation folds. Rounding is performed
    within each class.
    """
    by_label: dict[str, list[Example]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)
    if len(by_label) < 2:
        raise ValueError("A stratified classification split requires at least two labels")

    rng = random.Random(seed)
    pool: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []
    for label in sorted(by_label):
        members = list(by_label[label])
        rng.shuffle(members)
        n_pool = round(len(members) * 0.30)
        n_val = round(len(members) * 0.30)
        pool.extend(members[:n_pool])
        val.extend(members[n_pool : n_pool + n_val])
        test.extend(members[n_pool + n_val :])

    # Prevent class-block ordering from leaking into later sampling.
    rng.shuffle(pool)
    rng.shuffle(val)
    rng.shuffle(test)
    assigned = [example.id for part in (pool, val, test) for example in part]
    if len(assigned) != len(examples) or len(set(assigned)) != len(examples):
        raise AssertionError("Stratified split lost or duplicated examples")
    return Split(pool=pool, val=val, test=test)


def limit_split(
    split: Split,
    max_pool: int | None,
    max_val: int | None,
    max_test: int | None,
) -> Split:
    return Split(
        pool=split.pool[:max_pool] if max_pool else split.pool,
        val=split.val[:max_val] if max_val else split.val,
        test=split.test[:max_test] if max_test else split.test,
    )


def select_initial_demos(
    pool: list[Example], positive_label: str, seed: int
) -> list[Example]:
    positive = [example for example in pool if example.label == positive_label]
    negative = [example for example in pool if example.label != positive_label]
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("Initial prompt requires at least 2 positive and 2 negative examples")
    rng = random.Random(seed)
    demos = rng.sample(positive, 2) + rng.sample(negative, 2)
    rng.shuffle(demos)
    return demos
