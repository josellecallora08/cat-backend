"""Configuration and scheduler tests for quarantine cleanup."""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import Settings
from app.services import upload_quarantine
from app.services.upload_quarantine import CleanupResult, cleanup_expired_files


class TestQuarantineCleanupConfig:
    def test_negative_retention_is_zero_and_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.config"):
            settings = Settings(upload_quarantine_retention_hours=-1)

        assert settings.upload_quarantine_retention_hours == 0
        assert "treated as 0" in caplog.text

    def test_retention_above_limit_is_clamped(self):
        assert Settings(upload_quarantine_retention_hours=8761).upload_quarantine_retention_hours == 8760

    def test_interval_is_clamped_to_supported_range(self):
        assert Settings(upload_quarantine_cleanup_interval_minutes=0).upload_quarantine_cleanup_interval_minutes == 1
        assert Settings(upload_quarantine_cleanup_interval_minutes=1441).upload_quarantine_cleanup_interval_minutes == 1440

    def test_valid_values_are_not_changed(self):
        settings = Settings(
            upload_quarantine_retention_hours=48,
            upload_quarantine_cleanup_interval_minutes=15,
        )
        assert settings.upload_quarantine_retention_hours == 48
        assert settings.upload_quarantine_cleanup_interval_minutes == 15


async def test_scheduler_runs_an_initial_cleanup_and_can_be_cancelled():
    app = MagicMock()
    cleanup = AsyncMock()
    fake_session_factory = MagicMock()

    with patch("app.database.async_session_factory", fake_session_factory), \
         patch.object(upload_quarantine, "cleanup_expired_files", cleanup), \
         patch.object(upload_quarantine.settings, "upload_quarantine_cleanup_interval_minutes", 60):
        await upload_quarantine.start_cleanup_scheduler(app)
        cleanup.assert_awaited_once_with(fake_session_factory)
        app.state.quarantine_cleanup_task.cancel()
        try:
            await app.state.quarantine_cleanup_task
        except asyncio.CancelledError:
            pass

    cleanup.assert_awaited_once_with(fake_session_factory)


async def test_scheduler_reloads_interval_before_each_recurring_pass():
    app = MagicMock()
    cleanup = AsyncMock()
    fake_session_factory = MagicMock()
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 1:
            upload_quarantine.settings.upload_quarantine_cleanup_interval_minutes = 2
            return
        raise asyncio.CancelledError

    with patch("app.database.async_session_factory", fake_session_factory), \
         patch.object(upload_quarantine, "cleanup_expired_files", cleanup), \
         patch.object(upload_quarantine.asyncio, "sleep", side_effect=fake_sleep), \
         patch.object(
             upload_quarantine.settings,
             "upload_quarantine_cleanup_interval_minutes",
             1,
         ):
        await upload_quarantine.start_cleanup_scheduler(app)
        try:
            await app.state.quarantine_cleanup_task
        except asyncio.CancelledError:
            pass

    assert sleep_calls == [60, 120]
    assert cleanup.await_count == 2


async def test_next_cleanup_pass_uses_modified_retention_setting(tmp_path):
    source = tmp_path / "dynamic-retention.pdf"
    source.write_bytes(b"raw")
    one_hour_ago = time.time() - 3600
    source.touch()
    import os

    os.utime(source, (one_hour_ago, one_hour_ago))

    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=context)

    with patch.object(
        upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)
    ), patch.object(
        upload_quarantine.settings, "upload_quarantine_retention_hours", 2
    ):
        first = await cleanup_expired_files(session_factory)
        upload_quarantine.settings.upload_quarantine_retention_hours = 0
        second = await cleanup_expired_files(session_factory)

    assert first == CleanupResult()
    assert second == CleanupResult(orphaned=1)
    assert not source.exists()


async def test_scheduler_shutdown_timeout_does_not_block(caplog):
    app = MagicMock()
    release = asyncio.Event()

    async def stubborn_task():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(stubborn_task())
    await asyncio.sleep(0)
    app.state.quarantine_cleanup_task = task

    stopped = await upload_quarantine.stop_cleanup_scheduler(
        app, timeout_seconds=0.01
    )

    assert stopped is False
    assert "Timed out waiting" in caplog.text
    release.set()
    await task
