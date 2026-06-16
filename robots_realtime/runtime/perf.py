"""Lightweight per-interval performance accumulator for the node hot loops.

Each :class:`PerfStats` collects min / avg / max for any number of named
metrics and logs a one-line summary at most once per ``interval_s`` seconds.
Updates are O(1) and allocation-free on the hot path, so it is safe to call
``record()`` inside a 200 Hz (or flat-out) loop.

Logs go to the ``robots_realtime.perf`` logger, which in node subprocesses is
routed to that node's ``{name}.log`` file (see ProcessHost / _host_worker).

Enable/disable and tune via environment variables (read once at import):
    RR_PERF=0            disable all perf logging (default: enabled)
    RR_PERF_INTERVAL=5   seconds between summary lines (default: 2.0)
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("robots_realtime.perf")

_ENABLED = os.environ.get("RR_PERF", "1") != "0"
_DEFAULT_INTERVAL = float(os.environ.get("RR_PERF_INTERVAL", "2.0"))


class PerfStats:
    """Rolling min/avg/max accumulator that logs once per interval.

    Args:
        name:       Label prefix for log lines (typically the node name).
        interval_s: Seconds between emitted summary lines.
    """

    enabled = _ENABLED

    def __init__(self, name: str, interval_s: float = _DEFAULT_INTERVAL) -> None:
        self.name = name
        self.interval_s = float(interval_s)
        # label -> [count, sum, min, max]
        self._stats: dict[str, list[float]] = {}
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        """Reset the interval clock (call once the loop actually starts)."""
        self._stats.clear()
        self._t0 = time.perf_counter()

    def record(self, label: str, value: float) -> None:
        """Record one sample (any unit; ms by convention) for *label*."""
        if not self.enabled:
            return
        s = self._stats.get(label)
        if s is None:
            self._stats[label] = [1.0, value, value, value]
        else:
            s[0] += 1.0
            s[1] += value
            if value < s[2]:
                s[2] = value
            if value > s[3]:
                s[3] = value

    def maybe_log(self) -> None:
        """Emit a summary line if at least ``interval_s`` has elapsed."""
        if not self.enabled or not self._stats:
            return
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed < self.interval_s:
            return
        parts = []
        for label, (n, total, lo, hi) in self._stats.items():
            count = int(n)
            rate = count / elapsed if elapsed > 0 else 0.0
            parts.append(
                f"{label} avg={total / n:.2f} min={lo:.2f} max={hi:.2f} "
                f"(n={count}, {rate:.0f}/s)"
            )
        logger.info("[%s] %s", self.name, "  |  ".join(parts))
        self._stats.clear()
        self._t0 = now
