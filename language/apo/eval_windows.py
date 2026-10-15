"""Canonical eval windows aligned with run.py DataLoader (drop_last)."""

from __future__ import annotations

from typing import List

from data_provider.data_factory import data_provider


def loader_window_indices(args, split: str) -> List[int]:
    """Window indices seen by run.py / Exp test & val loops (batch_size + drop_last)."""
    _, loader = data_provider(args, split)
    seen: list[int] = []
    for batch in loader:
        index = batch[-1]
        index_np = index.numpy() if hasattr(index, "numpy") else index
        seen.extend(int(i) for i in index_np.tolist())
    return sorted(set(seen))


def dataset_window_indices(args, split: str) -> List[int]:
    """All windows in the dataset split (includes tail batch dropped by DataLoader)."""
    dataset, _ = data_provider(args, split)
    return list(range(len(dataset)))


def resolve_eval_window_indices(
    args,
    split: str,
    text: str,
    *,
    align_with_runpy: bool = True,
) -> List[int]:
    """Resolve window list for reporting MSE comparable to run.py metrics."""
    text = text.strip().lower()
    if text in {"runpy", "loader"}:
        return loader_window_indices(args, split)
    if text in {"all", "*"}:
        if align_with_runpy and split in ("test", "val"):
            return loader_window_indices(args, split)
        return dataset_window_indices(args, split)
    return [int(x.strip()) for x in text.split(",") if x.strip()]
