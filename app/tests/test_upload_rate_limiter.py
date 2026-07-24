"""Unit tests for app.services.upload_rate_limiter."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.services import upload_rate_limiter
from app.services.upload_rate_limiter import (
    cleanup_expired_entries,
    get_retry_after,
    is_rate_limited,
    record_rejection,
)


@pytest.fixture(autouse=True)
def clear_trackers():
    """Reset rate limiter state between tests."""
    upload_rate_limiter._rejection_tracker.clear()
    upload_rate_limiter._cooldown_tracker.clear()
    yield
    upload_rate_limiter._rejection_tracker.clear()
    upload_rate_limiter._cooldown_tracker.clear()


class TestRecordRejectionAndRateLimit:
    """Task 15.1 + 15.2: Test sliding window and rate limiting."""

    def test_under_limit_not_blocked(self):
        user = "user-a"
        for _ in range(5):  # limit is 10
            record_rejection(user)
        assert is_rate_limited(user) is False

    def test_at_limit_blocked(self):
        user = "user-b"
        for _ in range(10):  # exactly at limit
            record_rejection(user)
        assert is_rate_limited(user) is True

    def test_over_limit_blocked(self):
        user = "user-c"
        for _ in range(15):
            record_rejection(user)
        assert is_rate_limited(user) is True

    def test_no_rejections_not_blocked(self):
        assert is_rate_limited("new-user") is False


class TestCooldownEnforcement:
    """Task 15.3: Test cooldown is enforced after hitting limit."""

    def test_cooldown_activated_at_limit(self):
        user = "user-cooldown"
        for _ in range(10):
            record_rejection(user)
        # Cooldown should be set
        assert user in upload_rate_limiter._cooldown_tracker
        assert upload_rate_limiter._cooldown_tracker[user] > datetime.now(timezone.utc)

    def test_cooldown_blocks_even_with_empty_window(self):
        user = "user-cooldown-block"
        # Manually set a cooldown in the future
        upload_rate_limiter._cooldown_tracker[user] = datetime.now(timezone.utc) + timedelta(minutes=30)
        assert is_rate_limited(user) is True

    def test_expired_cooldown_does_not_block(self):
        user = "user-expired"
        # Set a cooldown in the past
        upload_rate_limiter._cooldown_tracker[user] = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert is_rate_limited(user) is False


class TestGetRetryAfter:
    """Task 15.2: Test Retry-After calculation."""

    def test_returns_zero_when_not_limited(self):
        assert get_retry_after("nobody") == 0

    def test_returns_cooldown_seconds(self):
        user = "user-retry"
        upload_rate_limiter._cooldown_tracker[user] = datetime.now(timezone.utc) + timedelta(minutes=15)
        retry = get_retry_after(user)
        # Should be approximately 15*60 = 900 seconds
        assert 890 <= retry <= 901

    def test_returns_window_expiry_when_at_limit(self):
        user = "user-window"
        # Add 10 rejections now
        for _ in range(10):
            record_rejection(user)
        retry = get_retry_after(user)
        # Should be approximately window_minutes * 60 seconds (oldest entry expires)
        assert retry > 0


class TestCleanupExpiredEntries:
    """Task 15.4: Test cleanup removes expired entries."""

    def test_removes_old_entries(self):
        user = "user-old"
        # Manually insert old timestamps
        old_time = datetime.now(timezone.utc) - timedelta(minutes=120)
        upload_rate_limiter._rejection_tracker[user] = [old_time]
        cleaned = cleanup_expired_entries()
        assert cleaned >= 1
        assert user not in upload_rate_limiter._rejection_tracker

    def test_keeps_recent_entries(self):
        user = "user-recent"
        record_rejection(user)
        cleanup_expired_entries()
        assert user in upload_rate_limiter._rejection_tracker

    def test_removes_expired_cooldowns(self):
        user = "user-expired-cd"
        upload_rate_limiter._cooldown_tracker[user] = datetime.now(timezone.utc) - timedelta(minutes=5)
        cleanup_expired_entries()
        assert user not in upload_rate_limiter._cooldown_tracker
