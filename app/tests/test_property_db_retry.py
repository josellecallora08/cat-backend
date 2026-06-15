"""Property-based tests for database retry mechanism.

Feature: collection-agent-trainer, Property 14: Database retry mechanism

For any database write operation that fails, the system SHALL retry up to exactly
3 times before reporting failure. If all retries are exhausted, the data SHALL
remain accessible in memory for the duration of the active session.

Validates: Requirements 8.3, 8.5
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy.exc import OperationalError

from app.services.db_retry import (
    MAX_RETRIES,
    DatabasePersistenceError,
    InMemoryFallbackStore,
    retry_db_operation,
)


# --- Strategies ---

# Number of consecutive failures before success (or never succeeding)
failure_counts = st.integers(min_value=1, max_value=10)

# Random data payloads to persist
data_payloads = st.one_of(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L", "N"))),
        values=st.one_of(
            st.text(min_size=0, max_size=100),
            st.integers(min_value=-10000, max_value=10000),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
        min_size=1,
        max_size=5,
    ),
    st.text(min_size=1, max_size=200),
    st.lists(st.integers(), min_size=1, max_size=10),
)

# Session IDs
session_ids = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(categories=("L", "N"), whitelist_characters="-_"),
)


# --- Property Tests ---


class TestRetryMechanismProperty:
    """Property 14: Database retry mechanism.

    Feature: collection-agent-trainer, Property 14: Database retry mechanism
    """

    @settings(max_examples=100)
    @given(num_failures=failure_counts)
    @pytest.mark.asyncio
    async def test_retry_count_and_outcome(self, num_failures: int):
        """**Validates: Requirements 8.3, 8.5**

        For any number of failures:
        - If failures < MAX_RETRIES (3): operation eventually succeeds
        - If failures >= MAX_RETRIES (3): exactly 3 attempts are made and
          DatabasePersistenceError is raised
        """
        call_count = 0
        db_error = OperationalError("stmt", {}, Exception("connection refused"))

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= num_failures:
                raise db_error
            return "success"

        # Reset call count for each test
        call_count = 0

        if num_failures < MAX_RETRIES:
            # Operation should eventually succeed
            result = await retry_db_operation(
                flaky_operation,
                session_id="test-sess",
                data={"payload": "test"},
                backoff_delays=[0, 0, 0],
            )
            assert result == "success"
            # Should have been called num_failures + 1 times (failures + 1 success)
            assert call_count == num_failures + 1
        else:
            # Operation should fail after exactly MAX_RETRIES attempts
            with pytest.raises(DatabasePersistenceError):
                await retry_db_operation(
                    flaky_operation,
                    session_id="test-sess",
                    data={"payload": "test"},
                    backoff_delays=[0, 0, 0],
                )
            # Should have been called exactly MAX_RETRIES times
            assert call_count == MAX_RETRIES

    @settings(max_examples=100)
    @given(data=data_payloads, session_id=session_ids)
    @pytest.mark.asyncio
    async def test_data_preserved_in_memory_on_exhaustion(self, data, session_id: str):
        """**Validates: Requirements 8.3, 8.5**

        For any data payload and session ID, when all retries are exhausted,
        the data SHALL remain accessible in memory via the fallback store.
        """
        db_error = OperationalError("stmt", {}, Exception("connection refused"))
        operation = AsyncMock(side_effect=db_error)

        # Use a fresh store to avoid cross-test interference
        store = InMemoryFallbackStore()

        with patch("app.services.db_retry.fallback_store", store):
            with pytest.raises(DatabasePersistenceError) as exc_info:
                await retry_db_operation(
                    operation,
                    session_id=session_id,
                    data=data,
                    backoff_delays=[0, 0, 0],
                )

            # Verify the error carries the session_id and data
            assert exc_info.value.session_id == session_id
            assert exc_info.value.data == data

            # Verify data is preserved in the fallback store
            assert store.has_data(session_id)
            stored_items = store.get(session_id)
            assert data in stored_items

    @settings(max_examples=100)
    @given(num_failures=st.integers(min_value=3, max_value=10))
    @pytest.mark.asyncio
    async def test_exactly_three_attempts_on_persistent_failure(self, num_failures: int):
        """**Validates: Requirements 8.3, 8.5**

        Regardless of how many times the operation would fail,
        the system SHALL attempt exactly 3 times (MAX_RETRIES) before giving up.
        """
        call_count = 0
        db_error = OperationalError("stmt", {}, Exception("persistent failure"))

        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise db_error

        call_count = 0

        with pytest.raises(DatabasePersistenceError):
            await retry_db_operation(
                always_failing,
                session_id="sess-exact",
                data={"test": True},
                backoff_delays=[0, 0, 0],
            )

        assert call_count == MAX_RETRIES, (
            f"Expected exactly {MAX_RETRIES} attempts, got {call_count}"
        )
