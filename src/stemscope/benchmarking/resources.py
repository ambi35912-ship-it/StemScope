"""Process-wide resource observations, not isolated model allocations."""

from threading import Event, Thread
from time import perf_counter

import psutil


class ResourceMonitor:
    """Sample this process RSS every 50 ms; CPU time covers all its threads."""

    def __enter__(self):
        self.process = psutil.Process()
        self.baseline_rss = self.peak_rss = self.process.memory_info().rss
        self.cpu_start = sum(self.process.cpu_times()[:2])
        self.start = perf_counter()
        self.stop = Event()
        self.thread = Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.05):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.seconds = perf_counter() - self.start
        self.cpu_seconds = sum(self.process.cpu_times()[:2]) - self.cpu_start
        self.cpu_percent = 100 * self.cpu_seconds / max(self.seconds, 1e-9)
