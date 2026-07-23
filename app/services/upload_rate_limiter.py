"""Upload rejection rate limiter with cooldown enforcement.

Tracks upload rejections per user in a sliding window and enforces
a cooldown period when the rejection threshold is exceeded.
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from app.config import settings

# Module-level state
_rejection_tracker: Dict[str, List[datetime]] = {}
_cooldown_tracker: Dict[str, datetime] = {}  # user_id → cooldown_expires_at
_tracker_lock = threading.Lock()


def record_rejection(user_id: str) -> None:
    """Record a rejection event for the given user.

    Appends the current UTC timestamp to the user's rejection history.
    If the user reaches or exceeds the maximum allowed attempts within
    the sliding window, a cooldown period is activated.

    Args:
        user_id: The unique identifier of the user.
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=settings.upload_rejection_window_minutes)

    with _tracker_lock:
        if user_id not in _rejection_tracker:
            _rejection_tracker[user_id] = []

        _rejection_tracker[user_id].append(now)

        # Count rejections within the sliding window
        recent = [t for t in _rejection_tracker[user_id] if t > window_start]
        _rejection_tracker[user_id] = recent

        # Activate cooldown if at or above threshold
        if len(recent) >= settings.upload_rejection_max_attempts:
            _cooldown_tracker[user_id] = now + timedelta(
                minutes=settings.upload_rejection_cooldown_minutes
            )


def is_rate_limited(user_id: str) -> bool:
    """Check whether the user is currently rate limited.

    A user is rate limited if:
    - They have an active cooldown (cooldown expiry is in the future), OR
    - Their rejection count within the sliding window meets or exceeds the threshold.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        True if the user is rate limited, False otherwise.
    """
    now = datetime.now(timezone.utc)

    with _tracker_lock:
        # Check cooldown first
        if user_id in _cooldown_tracker:
            if _cooldown_tracker[user_id] > now:
                return True

        # Check sliding window
        if user_id in _rejection_tracker:
            window_start = now - timedelta(
                minutes=settings.upload_rejection_window_minutes
            )
            recent = [t for t in _rejection_tracker[user_id] if t > window_start]
            if len(recent) >= settings.upload_rejection_max_attempts:
                return True

        return False


def get_retry_after(user_id: str) -> int:
    """Get the number of seconds until the user can retry.

    If the user is in cooldown, returns seconds until cooldown expires.
    If the user is at the sliding window limit, returns seconds until
    the oldest entry in the window expires (freeing a slot).
    Otherwise returns 0.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        Number of seconds until the rate limit expires, or 0 if not limited.
    """
    now = datetime.now(timezone.utc)

    with _tracker_lock:
        # Check cooldown first
        if user_id in _cooldown_tracker:
            expires_at = _cooldown_tracker[user_id]
            if expires_at > now:
                return int((expires_at - now).total_seconds()) + 1

        # Check sliding window
        if user_id in _rejection_tracker:
            window_start = now - timedelta(
                minutes=settings.upload_rejection_window_minutes
            )
            recent = sorted(t for t in _rejection_tracker[user_id] if t > window_start)
            if len(recent) >= settings.upload_rejection_max_attempts:
                # Oldest entry in window — when it expires, a slot opens
                oldest = recent[0]
                expires_at = oldest + timedelta(
                    minutes=settings.upload_rejection_window_minutes
                )
                return int((expires_at - now).total_seconds()) + 1

        return 0


def cleanup_expired_entries() -> int:
    """Remove expired rejection records and cooldowns.

    Removes entries older than the sliding window from the rejection tracker,
    removes expired cooldowns, and removes users with empty rejection lists.

    Returns:
        Count of users fully cleaned up (removed from both trackers).
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=settings.upload_rejection_window_minutes)
    cleaned_count = 0

    with _tracker_lock:
        # Clean up rejection tracker
        users_to_remove: List[str] = []
        for user_id in list(_rejection_tracker.keys()):
            _rejection_tracker[user_id] = [
                t for t in _rejection_tracker[user_id] if t > window_start
            ]
            if not _rejection_tracker[user_id]:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del _rejection_tracker[user_id]

        # Clean up expired cooldowns
        expired_cooldowns: List[str] = []
        for user_id, expires_at in _cooldown_tracker.items():
            if expires_at <= now:
                expired_cooldowns.append(user_id)

        for user_id in expired_cooldowns:
            del _cooldown_tracker[user_id]

        # Count users fully cleaned (removed from rejection tracker)
        # that are also no longer in cooldown
        for user_id in users_to_remove:
            if user_id not in _cooldown_tracker:
                cleaned_count += 1

    return cleaned_count
