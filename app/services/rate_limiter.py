"""In-memory rate limiter for password reset endpoints.

Uses a sliding window approach. For production, swap with Redis-backed implementation.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RateLimitWindow:
    """Tracks request timestamps within a window."""
    timestamps: list[float] = field(default_factory=list)


class RateLimiter:
    """Thread-safe in-memory rate limiter with sliding window."""

    def __init__(self):
        self._store: dict[str, RateLimitWindow] = defaultdict(RateLimitWindow)
        self._lock = threading.Lock()

    def is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if a key has exceeded its rate limit.

        Args:
            key: Unique identifier (e.g., "forgot:ip:1.2.3.4" or "forgot:email:user@test.com")
            max_requests: Maximum allowed requests in the window
            window_seconds: Window duration in seconds

        Returns:
            True if rate limited, False if allowed
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            window = self._store[key]
            # Remove expired timestamps
            window.timestamps = [ts for ts in window.timestamps if ts > cutoff]

            if len(window.timestamps) >= max_requests:
                return True

            # Record this request
            window.timestamps.append(now)
            return False

    def cleanup(self) -> None:
        """Remove stale entries. Call periodically in production."""
        now = time.time()
        max_age = 3600  # 1 hour

        with self._lock:
            keys_to_remove = []
            for key, window in self._store.items():
                if not window.timestamps or (now - max(window.timestamps)) > max_age:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._store[key]


# Global instance
rate_limiter = RateLimiter()


# Rate limit configuration
FORGOT_PASSWORD_IP_LIMIT = 5  # max requests per IP
FORGOT_PASSWORD_IP_WINDOW = 900  # 15 minutes

FORGOT_PASSWORD_EMAIL_LIMIT = 3  # max requests per email
FORGOT_PASSWORD_EMAIL_WINDOW = 3600  # 1 hour

RESET_PASSWORD_IP_LIMIT = 5  # max reset attempts per IP
RESET_PASSWORD_IP_WINDOW = 900  # 15 minutes

RESET_PASSWORD_TOKEN_LIMIT = 3  # max attempts per token hash
RESET_PASSWORD_TOKEN_WINDOW = 900  # 15 minutes
