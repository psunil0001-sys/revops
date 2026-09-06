"""Bounded parallelism helper for multi-fund / analyst work queues."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_queued(
    items: list[T],
    fn: Callable[[T], R],
    *,
    max_parallel: int = 1,
) -> list[tuple[T, R | BaseException]]:
    """Run ``fn`` over ``items`` with at most ``max_parallel`` workers.

    Returns one ``(item, result_or_exception)`` per input, in the same order
    as ``items``. Exceptions from ``fn`` are captured, not raised.
    """
    if not items:
        return []
    workers = max(1, min(int(max_parallel), len(items)))
    by_index: dict[int, tuple[T, R | BaseException]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            item = items[idx]
            try:
                by_index[idx] = (item, future.result())
            except BaseException as exc:  # noqa: BLE001
                by_index[idx] = (item, exc)

    return [by_index[i] for i in range(len(items))]
