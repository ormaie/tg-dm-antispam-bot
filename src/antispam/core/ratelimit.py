from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter to protect Telegram accounts from burst actions."""

    def __init__(
        self,
        max_events: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._max_events = max_events
        self._window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._timestamps: deque[float] = deque()

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def allow(self) -> bool:
        now = self._clock()
        self._evict_expired(now)
        if len(self._timestamps) >= self._max_events:
            return False
        self._timestamps.append(now)
        return True

    def remaining(self) -> int:
        now = self._clock()
        self._evict_expired(now)
        return max(0, self._max_events - len(self._timestamps))
