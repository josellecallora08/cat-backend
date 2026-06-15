"""Database retry wrapper with exponential backoff.

Provides a decorator/utility that retries failed database write operations
up to 3 times with exponential backoff (1s, 2s, 4s). On exhaustion, data
is preserved in memory and a notification-level exception is raised.

Validates: Requirements 8.3, 8.5
"""

import asyncio
import functools
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BACKOFF_DELAYS = [1.0, 2.0, 4.0]  # seconds between attempts


class DatabasePersistenceError(Exception):
    """Raised when all database retry attempts are exhausted.

    This is a notification-level exception indicating that data could not
    be persisted but has been preserved in memory for the session duration.
    """

    def __init__(self, message: str, session_id: str | None = None, data: Any = None):
        self.session_id = session_id
        self.data = data
        super().__init__(message)


class InMemoryFallbackStore:
    """Stores data that failed to persist to the database.

    Data is keyed by session_id so it can be recovered later during the
    active session lifetime.
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[Any]] = defaultdict(list)

    def save(self, session_id: str, data: Any) -> None:
        """Save data to the in-memory store for later recovery."""
        self._store[session_id].append(data)
        logger.warning(
            "Data saved to in-memory fallback store for session %s", session_id
        )

    def get(self, session_id: str) -> List[Any]:
        """Retrieve all stored data for a given session."""
        return self._store.get(session_id, [])

    def has_data(self, session_id: str) -> bool:
        """Check if there is any stored data for a session."""
        return session_id in self._store and len(self._store[session_id]) > 0

    def clear(self, session_id: str) -> None:
        """Clear stored data for a session (e.g., after successful recovery)."""
        if session_id in self._store:
            del self._store[session_id]

    def all_sessions(self) -> List[str]:
        """Return all session IDs that have stored data."""
        return list(self._store.keys())


# Module-level fallback store instance
fallback_store = InMemoryFallbackStore()


async def retry_db_operation(
    operation: Callable,
    *args: Any,
    session_id: str | None = None,
    data: Any = None,
    max_retries: int = MAX_RETRIES,
    backoff_delays: list[float] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a database operation with retry and exponential backoff.

    Args:
        operation: The async callable to execute (database write operation).
        *args: Positional arguments to pass to the operation.
        session_id: The session ID for fallback data storage.
        data: The data being written (preserved in memory on failure).
        max_retries: Maximum number of retry attempts (default: 3).
        backoff_delays: List of delay durations between retries in seconds.
        **kwargs: Keyword arguments to pass to the operation.

    Returns:
        The result of the successful operation.

    Raises:
        DatabasePersistenceError: When all retries are exhausted.
    """
    if backoff_delays is None:
        backoff_delays = BACKOFF_DELAYS

    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            result = await operation(*args, **kwargs)
            return result
        except (SQLAlchemyError, OSError) as exc:
            last_exception = exc
            logger.warning(
                "Database operation failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                str(exc),
            )

            # Wait before retrying (except after the last attempt)
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                await asyncio.sleep(delay)

    # All retries exhausted - preserve data in memory
    if session_id and data is not None:
        fallback_store.save(session_id, data)

    raise DatabasePersistenceError(
        f"Database write failed after {max_retries} attempts: {last_exception}",
        session_id=session_id,
        data=data,
    )


def with_db_retry(
    session_id_param: str | None = None,
    data_param: str | None = None,
    max_retries: int = MAX_RETRIES,
    backoff_delays: list[float] | None = None,
):
    """Decorator that adds database retry logic to an async function.

    Args:
        session_id_param: Name of the kwarg/arg that contains the session_id.
        data_param: Name of the kwarg/arg that contains the data to preserve.
        max_retries: Maximum number of retry attempts.
        backoff_delays: List of delay durations between retries.

    Usage:
        @with_db_retry(session_id_param="session_id", data_param="entry")
        async def save_transcript(session_id: str, entry: dict, db: AsyncSession):
            ...
    """
    if backoff_delays is None:
        backoff_delays = BACKOFF_DELAYS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract session_id and data from function arguments
            session_id = _extract_param(func, session_id_param, args, kwargs)
            data = _extract_param(func, data_param, args, kwargs)

            return await retry_db_operation(
                func,
                *args,
                session_id=str(session_id) if session_id else None,
                data=data,
                max_retries=max_retries,
                backoff_delays=backoff_delays,
                **kwargs,
            )

        return wrapper

    return decorator


def _extract_param(
    func: Callable, param_name: str | None, args: tuple, kwargs: dict
) -> Any:
    """Extract a parameter value from function arguments by name."""
    if param_name is None:
        return None

    # Check kwargs first
    if param_name in kwargs:
        return kwargs[param_name]

    # Try to find by positional index using function signature
    import inspect

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if param_name in params:
        idx = params.index(param_name)
        if idx < len(args):
            return args[idx]

    return None
