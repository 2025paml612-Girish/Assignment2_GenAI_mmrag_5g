"""DoS protection: payload size limit, per-session token-bucket rate limit,
burst-repeat detection and a global concurrency cap. All checks are O(1)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class DoSGuard:
    def __init__(self, max_chars: int = 1000, rpm: int = 30, max_concurrent: int = 4, burst_repeat_limit: int = 5):
        self.max_chars = max_chars
        self.rpm = rpm
        self.burst = burst_repeat_limit
        self.sem = threading.BoundedSemaphore(max_concurrent)
        self.buckets: dict[str, list[float]] = defaultdict(lambda: [float(rpm), time.monotonic()])
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.lock = threading.Lock()

    def check(self, session: str, query: str) -> tuple[bool, str]:
        if len(query) > self.max_chars:
            return False, f"dos: query longer than {self.max_chars} characters"
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets[session]
            tokens = min(self.rpm, tokens + (now - last) * self.rpm / 60.0)
            if tokens < 1:
                self.buckets[session] = [tokens, now]
                return False, "dos: rate limit exceeded"
            self.buckets[session] = [tokens - 1, now]
            rec = self.recent[session]
            key = query.strip().lower()
            n_same = sum(1 for t, q in rec if q == key and now - t < 10)
            rec.append((now, key))
            if n_same >= self.burst:
                return False, "dos: repeated query burst"
        return True, ""

    def acquire(self) -> bool:
        return self.sem.acquire(timeout=5)

    def release(self) -> None:
        try:
            self.sem.release()
        except ValueError:
            pass
