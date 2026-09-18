from time import perf_counter


def elapsed(start: float) -> float:
    """Return wall-clock seconds since a perf_counter timestamp."""
    return perf_counter() - start
