"""Tests for database retry wrapper with exponential backoff.

Validates: Requirements 8.3, 8.5
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from app.services.db_retry import (
    MAX_RETRIES,
    DatabasePersistenceError,
    InMemoryFallbackStore,
    fallback_store,
    retry_db_operation,
    with_db_retry,
)


class TestDatabasePersistenceError:
    """Tests for the custom exception class."""

    def test_error_carries_session_id(self):
        err = DatabasePersistenceError("fail", session_id="sess-1", data={"key": "val"})
        assert err.session_id == "sess-1"
        assert err.data == {"key": "val"}
        assert "fail" in str(err)

    def test_error_without_optional_fields(self):
        err = DatabasePersistenceError("simple failure")
        assert err.session_id is None
        assert err.data is None


class TestInMemoryFallbackStore:
    """Tests for the in-memory fallback store."""

    def test_save_and_retrieve(self):
        store = InMemoryFallbackStore()
        store.save("sess-1", {"text": "hello"})
        assert store.get("sess-1") == [{"text": "hello"}]

    def test_save_multiple_items(self):
        store = InMemoryFallbackStore()
        store.save("sess-1", {"a": 1})
        store.save("sess-1", {"b": 2})
        assert len(store.get("sess-1")) == 2

    def test_get_empty_session(self):
        store = InMemoryFallbackStore()
        assert store.get("nonexistent") == []

    def test_has_data(self):
        store = InMemoryFallbackStore()
        assert not store.has_data("sess-1")
        store.save("sess-1", "data")
        assert store.has_data("sess-1")

    def test_clear(self):
        store = InMemoryFallbackStore()
        store.save("sess-1", "data")
        store.clear("sess-1")
        assert not store.has_data("sess-1")

    def test_all_sessions(self):
        store = InMemoryFallbackStore()
        store.save("sess-1", "a")
        store.save("sess-2", "b")
        sessions = store.all_sessions()
        assert "sess-1" in sessions
        assert "sess-2" in sessions


class TestRetryDbOperation:
    """Tests for the retry_db_operation utility function."""

    async def test_succeeds_on_first_attempt(self):
        operation = AsyncMock(return_value="success")

        result = await retry_db_operation(
            operation, session_id="sess-1", data="payload", backoff_delays=[0, 0, 0]
        )

        assert result == "success"
        assert operation.call_count == 1

    async def test_succeeds_after_retry(self):
        operation = AsyncMock(
            side_effect=[
                OperationalError("stmt", {}, Exception("conn refused")),
                "success",
            ]
        )

        result = await retry_db_operation(
            operation, session_id="sess-1", data="payload", backoff_delays=[0, 0, 0]
        )

        assert result == "success"
        assert operation.call_count == 2

    async def test_retries_exactly_3_times(self):
        db_error = OperationalError("stmt", {}, Exception("conn refused"))
        operation = AsyncMock(side_effect=[db_error, db_error, db_error])

        with pytest.raises(DatabasePersistenceError) as exc_info:
            await retry_db_operation(
                operation,
                session_id="sess-1",
                data={"important": "data"},
                backoff_delays=[0, 0, 0],
            )

        assert operation.call_count == MAX_RETRIES
        assert exc_info.value.session_id == "sess-1"
        assert exc_info.value.data == {"important": "data"}

    async def test_preserves_data_in_memory_on_exhaustion(self):
        # Use a fresh fallback store to avoid interference
        db_error = OperationalError("stmt", {}, Exception("conn refused"))
        operation = AsyncMock(side_effect=[db_error, db_error, db_error])

        # Clear any previous state
        fallback_store.clear("sess-retry-test")

        with pytest.raises(DatabasePersistenceError):
            await retry_db_operation(
                operation,
                session_id="sess-retry-test",
                data={"preserved": True},
                backoff_delays=[0, 0, 0],
            )

        assert fallback_store.has_data("sess-retry-test")
        stored = fallback_store.get("sess-retry-test")
        assert {"preserved": True} in stored

        # Clean up
        fallback_store.clear("sess-retry-test")

    async def test_exponential_backoff_delays(self):
        """Verify that delays follow exponential backoff pattern."""
        db_error = OperationalError("stmt", {}, Exception("conn refused"))
        operation = AsyncMock(side_effect=[db_error, db_error, db_error])

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with patch("app.services.db_retry.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(DatabasePersistenceError):
                await retry_db_operation(
                    operation,
                    session_id="sess-1",
                    data="x",
                    backoff_delays=[1.0, 2.0, 4.0],
                )

        # Should sleep between retries (not after last attempt)
        assert sleep_calls == [1.0, 2.0]

    async def test_does_not_retry_non_db_errors(self):
        """Non-SQLAlchemy/OS errors should not be retried."""
        operation = AsyncMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            await retry_db_operation(
                operation, session_id="sess-1", data="x", backoff_delays=[0, 0, 0]
            )

        assert operation.call_count == 1

    async def test_passes_args_and_kwargs(self):
        operation = AsyncMock(return_value="ok")

        result = await retry_db_operation(
            operation, "arg1", "arg2", session_id="s", data=None, key="val"
        )

        assert result == "ok"
        operation.assert_called_once_with("arg1", "arg2", key="val")


class TestWithDbRetryDecorator:
    """Tests for the decorator version of the retry wrapper."""

    async def test_decorator_retries_and_succeeds(self):
        call_count = 0

        @with_db_retry(
            session_id_param="session_id",
            data_param="entry",
            backoff_delays=[0, 0, 0],
        )
        async def save_entry(session_id: str, entry: dict):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OperationalError("stmt", {}, Exception("fail"))
            return "saved"

        result = await save_entry("sess-1", {"text": "hi"})
        assert result == "saved"
        assert call_count == 2

    async def test_decorator_raises_after_exhaustion(self):
        @with_db_retry(
            session_id_param="session_id",
            data_param="data",
            backoff_delays=[0, 0, 0],
        )
        async def always_fails(session_id: str, data: dict):
            raise OperationalError("stmt", {}, Exception("always"))

        with pytest.raises(DatabasePersistenceError) as exc_info:
            await always_fails("sess-dec", {"key": "value"})

        assert exc_info.value.session_id == "sess-dec"
        assert exc_info.value.data == {"key": "value"}

    async def test_decorator_stores_in_fallback_on_exhaustion(self):
        fallback_store.clear("sess-dec-store")

        @with_db_retry(
            session_id_param="session_id",
            data_param="payload",
            backoff_delays=[0, 0, 0],
        )
        async def fail_write(session_id: str, payload: dict):
            raise OperationalError("stmt", {}, Exception("no conn"))

        with pytest.raises(DatabasePersistenceError):
            await fail_write("sess-dec-store", {"saved_in_mem": True})

        assert fallback_store.has_data("sess-dec-store")
        assert {"saved_in_mem": True} in fallback_store.get("sess-dec-store")

        fallback_store.clear("sess-dec-store")
