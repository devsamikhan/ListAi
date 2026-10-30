"""High-resolution timing and memory measurement utilities using standard library."""

from contextlib import contextmanager
import statistics
import time
import tracemalloc
from typing import Callable, Dict, Generator, List, Tuple, TypeVar

T = TypeVar("T")


@contextmanager
def timer() -> Generator[Dict[str, float], None, None]:
    """Context manager measuring elapsed wall-clock time in seconds via perf_counter."""
    record: Dict[str, float] = {"elapsed_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed_seconds"] = time.perf_counter() - start


def measure_repeated(
    func: Callable[[], T],
    runs: int = 5,
    measure_memory: bool = False,
) -> Tuple[T, Dict[str, float]]:
    """Execute `func` multiple times and summarize timing (and optional peak memory).

    Returns the result of the final run and a dictionary of summary statistics.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1.")

    timings: List[float] = []
    peak_memories_kb: List[float] = []
    last_result: T

    for _ in range(runs):
        if measure_memory:
            tracemalloc.start()
        start = time.perf_counter()
        last_result = func()
        elapsed = time.perf_counter() - start
        if measure_memory:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memories_kb.append(peak_bytes / 1024.0)
        timings.append(elapsed)

    stats: Dict[str, float] = {
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.mean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "runs": float(runs),
    }
    if measure_memory and peak_memories_kb:
        stats["median_peak_kb"] = statistics.median(peak_memories_kb)
        stats["max_peak_kb"] = max(peak_memories_kb)

    return last_result, stats
