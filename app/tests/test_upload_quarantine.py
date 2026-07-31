"""Unit tests for app.services.upload_quarantine."""

import platform
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import upload_quarantine
from app.services.upload_quarantine import (
    CleanupResult,
    cleanup_expired_files,
    get_quarantine_path,
    sanitize_filename,
    store_in_quarantine,
)


class TestGetQuarantinePath:
    """Task 14.3: Test quarantine directory creation and permissions."""

    def test_creates_directory_if_missing(self, tmp_path):
        qdir = tmp_path / "new_quarantine"
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(qdir)):
            result = get_quarantine_path()
            assert result.exists()
            assert result.is_dir()

    def test_returns_absolute_path(self, tmp_path):
        qdir = tmp_path / "quarantine"
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(qdir)):
            result = get_quarantine_path()
            assert result.is_absolute()

    def test_idempotent_when_exists(self, tmp_path):
        qdir = tmp_path / "existing"
        qdir.mkdir()
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(qdir)):
            result = get_quarantine_path()
            assert result.exists()

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="POSIX permissions not enforced on Windows"
    )
    def test_restricted_permissions(self, tmp_path):
        import stat
        qdir = tmp_path / "perms_test"
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(qdir)):
            result = get_quarantine_path()
            mode = stat.S_IMODE(result.stat().st_mode)
            assert mode == 0o700


class TestStoreInQuarantine:
    """Task 14.1: Test UUID filename generation and storage."""

    def test_stores_file_with_uuid_name(self, tmp_path):
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)):
            content = b"test file content"
            path = store_in_quarantine(content, ".pdf")
            assert path.exists()
            assert path.read_bytes() == content
            # UUID filename pattern: 8-4-4-4-12 hex chars + extension
            stem = path.stem
            assert len(stem) == 36  # UUID4 string length
            assert path.suffix == ".pdf"

    def test_stores_different_extensions(self, tmp_path):
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)):
            path_md = store_in_quarantine(b"# markdown", ".md")
            path_csv = store_in_quarantine(b"a,b,c", ".csv")
            assert path_md.suffix == ".md"
            assert path_csv.suffix == ".csv"
            assert path_md.stem != path_csv.stem  # Different UUIDs


class TestCleanupExpiredFiles:
    """Quarantine cleanup retains audit records while removing raw files."""

    @staticmethod
    def _session_factory(upload=None, commit_error=None):
        session = AsyncMock()
        db_result = MagicMock()
        db_result.scalar_one_or_none.return_value = upload
        session.execute = AsyncMock(return_value=db_result)
        session.commit = AsyncMock(side_effect=commit_error)

        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=session)
        context.__aexit__ = AsyncMock(return_value=False)
        return MagicMock(return_value=context), session

    async def test_deletes_expired_file_after_recording_deleted_at(self, tmp_path):
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)):
            with patch.object(upload_quarantine.settings, "upload_quarantine_retention_hours", 24):
                old_file = tmp_path / "matching.pdf"
                old_file.write_bytes(b"old")
                import os
                old_time = time.time() - (25 * 3600)
                os.utime(old_file, (old_time, old_time))

                upload = SimpleNamespace(
                    id="upload-id",
                    deleted_at=None,
                    content_hash="a" * 64,
                    extracted_content="sanitized text",
                    scan_status="clean",
                    scan_signature=None,
                    script_id="script-id",
                )
                session_factory, session = self._session_factory(upload)
                result = await cleanup_expired_files(session_factory)

                assert result == CleanupResult(deleted=1)
                assert not old_file.exists()
                assert upload.content_hash == "a" * 64
                assert upload.extracted_content == "sanitized text"
                assert upload.scan_status == "clean"
                assert upload.script_id == "script-id"
                session.commit.assert_awaited_once()
                assert session.execute.await_count == 2

    async def test_keeps_recent_files(self, tmp_path):
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)):
            with patch.object(upload_quarantine.settings, "upload_quarantine_retention_hours", 24):
                recent_file = tmp_path / "recent.pdf"
                recent_file.write_bytes(b"recent")

                session_factory, session = self._session_factory()
                result = await cleanup_expired_files(session_factory)
                assert result == CleanupResult()
                assert recent_file.exists()
                session.execute.assert_not_awaited()

    async def test_db_failure_preserves_file_for_retry(self, tmp_path, caplog):
        file_path = tmp_path / "retry.pdf"
        file_path.write_bytes(b"retry")
        import os
        old_time = time.time() - 3600
        os.utime(file_path, (old_time, old_time))
        upload = SimpleNamespace(id="upload-id", deleted_at=None)
        session_factory, _ = self._session_factory(upload, RuntimeError("database down"))

        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)), \
             patch.object(upload_quarantine.settings, "upload_quarantine_retention_hours", 0):
            result = await cleanup_expired_files(session_factory)

        assert result == CleanupResult(skipped=1, errors=1)
        assert file_path.exists()
        assert "Failed to record quarantine deletion for retry.pdf" in caplog.text

    async def test_deletes_orphan_and_skips_temp_and_gitkeep(self, tmp_path, caplog):
        orphan = tmp_path / "orphan.pdf"
        temp = tmp_path / ".tmp_upload_active"
        gitkeep = tmp_path / ".gitkeep"
        for file_path in (orphan, temp, gitkeep):
            file_path.write_bytes(b"content")
        import os
        old_time = time.time() - 3600
        for file_path in (orphan, temp, gitkeep):
            os.utime(file_path, (old_time, old_time))
        session_factory, _ = self._session_factory()

        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)), \
             patch.object(upload_quarantine.settings, "upload_quarantine_retention_hours", 0):
            result = await cleanup_expired_files(session_factory)

        assert result == CleanupResult(orphaned=1)
        assert not orphan.exists()
        assert temp.exists()
        assert gitkeep.exists()
        assert "Removed orphaned quarantine file: orphan.pdf" in caplog.text

    async def test_filesystem_error_does_not_stop_remaining_files(
        self, tmp_path, caplog
    ):
        blocked = tmp_path / "blocked.pdf"
        removable = tmp_path / "removable.pdf"
        for file_path in (blocked, removable):
            file_path.write_bytes(b"content")
        import os
        old_time = time.time() - 3600
        for file_path in (blocked, removable):
            os.utime(file_path, (old_time, old_time))

        session_factory, _ = self._session_factory()
        real_unlink = Path.unlink

        def selective_unlink(path, *args, **kwargs):
            if path.name == "blocked.pdf":
                raise PermissionError("access denied")
            return real_unlink(path, *args, **kwargs)

        with patch.object(
            upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)
        ), patch.object(
            upload_quarantine.settings, "upload_quarantine_retention_hours", 0
        ), patch.object(
            Path, "unlink", selective_unlink
        ):
            result = await cleanup_expired_files(session_factory)

        assert result == CleanupResult(orphaned=1, skipped=1, errors=1)
        assert blocked.exists()
        assert not removable.exists()
        assert "Failed to delete quarantine file blocked.pdf" in caplog.text

    async def test_file_removed_concurrently_is_logged_and_skipped(
        self, tmp_path, caplog
    ):
        source = tmp_path / "raced.pdf"
        source.write_bytes(b"content")
        import os
        old_time = time.time() - 3600
        os.utime(source, (old_time, old_time))

        session_factory, _ = self._session_factory()
        real_unlink = Path.unlink

        def raced_unlink(path, *args, **kwargs):
            real_unlink(path, *args, **kwargs)
            raise FileNotFoundError("removed by another cleanup pass")

        with patch.object(
            upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)
        ), patch.object(
            upload_quarantine.settings, "upload_quarantine_retention_hours", 0
        ), patch.object(
            Path, "unlink", raced_unlink
        ):
            result = await cleanup_expired_files(session_factory)

        assert result == CleanupResult(skipped=1)
        assert not source.exists()
        assert "Quarantine file already removed: raced.pdf" in caplog.text


class TestSanitizeFilename:
    """Task 14.4: Test filename sanitization."""

    def test_removes_path_traversal(self):
        assert "/" not in sanitize_filename("../../etc/passwd")
        assert "\\" not in sanitize_filename("..\\..\\windows\\system32")

    def test_removes_dotdot(self):
        result = sanitize_filename("..file..txt")
        assert ".." not in result

    def test_removes_null_bytes(self):
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_removes_control_chars(self):
        result = sanitize_filename("file\x01\x02\x03name.txt")
        assert all(ord(c) >= 0x20 for c in result)

    def test_collapses_multiple_dots(self):
        result = sanitize_filename("file....txt")
        assert "..." not in result

    def test_limits_length(self):
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_normal_filename_unchanged(self):
        assert sanitize_filename("report_2024.pdf") == "report_2024.pdf"
