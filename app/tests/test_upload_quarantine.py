"""Unit tests for app.services.upload_quarantine."""

import platform
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import upload_quarantine
from app.services.upload_quarantine import (
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
    """Task 14.2: Test cleanup deletes old files and keeps recent ones."""

    def test_deletes_old_files(self, tmp_path):
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)):
            with patch.object(upload_quarantine.settings, "upload_quarantine_retention_hours", 24):
                # Create a file and set its mtime to 25 hours ago
                old_file = tmp_path / "old.pdf"
                old_file.write_bytes(b"old")
                import os
                old_time = time.time() - (25 * 3600)
                os.utime(old_file, (old_time, old_time))

                deleted = cleanup_expired_files()
                assert deleted == 1
                assert not old_file.exists()

    def test_keeps_recent_files(self, tmp_path):
        with patch.object(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)):
            with patch.object(upload_quarantine.settings, "upload_quarantine_retention_hours", 24):
                recent_file = tmp_path / "recent.pdf"
                recent_file.write_bytes(b"recent")
                # File was just created, so mtime is now

                deleted = cleanup_expired_files()
                assert deleted == 0
                assert recent_file.exists()


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
