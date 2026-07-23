"""Unit tests for app.services.upload_config validation utilities."""

import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import upload_config
from app.services.upload_config import (
    UploadRejectionReason,
    cleanup_expired_entries,
    get_quarantine_path,
    is_rate_limited,
    record_rejection,
    sanitize_filename,
    validate_archive,
    validate_file_size,
    validate_file_type,
)


@pytest.fixture(autouse=True)
def clear_rejection_tracker():
    """Ensure the in-memory rejection tracker is clean before/after each test."""
    upload_config._rejection_tracker.clear()
    yield
    upload_config._rejection_tracker.clear()


# --- validate_file_type ---


class TestValidateFileType:
    @pytest.mark.parametrize(
        "filename,content_type",
        [
            ("report.pdf", "application/pdf"),
            (
                "doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            (
                "sheet.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            ("data.csv", "text/csv"),
            ("notes.txt", "text/plain"),
            ("audio.mp3", "audio/mpeg"),
            ("audio.wav", "audio/wav"),
            ("audio.wav", "audio/x-wav"),
            ("archive.zip", "application/zip"),
            ("archive.zip", "application/x-zip-compressed"),
        ],
    )
    def test_valid_extension_and_mime_combinations(self, filename, content_type):
        valid, reason = validate_file_type(filename, content_type)
        assert valid is True
        assert reason is None

    @pytest.mark.parametrize(
        "filename,content_type",
        [
            ("malware.exe", "application/octet-stream"),
            ("script.sh", "text/x-shellscript"),
            ("image.png", "image/png"),
            ("noext", "application/octet-stream"),
        ],
    )
    def test_invalid_extension_rejected(self, filename, content_type):
        valid, reason = validate_file_type(filename, content_type)
        assert valid is False
        assert reason == UploadRejectionReason.INVALID_EXTENSION

    @pytest.mark.parametrize(
        "filename,content_type",
        [
            # Spoofed extension: .pdf name but executable MIME type
            ("fake.pdf", "application/x-msdownload"),
            # .docx name but plain text MIME type
            ("fake.docx", "text/plain"),
            # .zip name but pdf MIME type
            ("fake.zip", "application/pdf"),
            # .txt name but zip MIME type
            ("fake.txt", "application/zip"),
        ],
    )
    def test_mime_extension_mismatch_rejected(self, filename, content_type):
        valid, reason = validate_file_type(filename, content_type)
        assert valid is False
        assert reason == UploadRejectionReason.MIME_EXTENSION_MISMATCH

    def test_extension_case_insensitivity(self):
        valid, reason = validate_file_type("REPORT.PDF", "application/pdf")
        assert valid is True
        assert reason is None


# --- validate_file_size ---


class TestValidateFileSize:
    def test_file_under_limit_is_valid(self):
        valid, reason = validate_file_size(1_000_000)  # ~1MB
        assert valid is True
        assert reason is None

    def test_file_exactly_at_limit_is_valid(self):
        valid, reason = validate_file_size(10_485_760)  # exactly 10 MB
        assert valid is True
        assert reason is None

    def test_file_one_byte_over_limit_is_rejected(self):
        valid, reason = validate_file_size(10_485_761)  # 10 MB + 1 byte
        assert valid is False
        assert reason == UploadRejectionReason.FILE_TOO_LARGE

    def test_file_well_over_limit_is_rejected(self):
        valid, reason = validate_file_size(50_000_000)
        assert valid is False
        assert reason == UploadRejectionReason.FILE_TOO_LARGE

    def test_zero_size_file_is_valid(self):
        valid, reason = validate_file_size(0)
        assert valid is True
        assert reason is None


# --- validate_archive ---


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    """Helper to create a zip file with given entries {name: content}."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return path


class TestValidateArchive:
    def test_valid_archive_within_limits(self, tmp_path):
        zip_path = _make_zip(
            tmp_path / "valid.zip",
            {"file1.txt": b"hello", "file2.txt": b"world"},
        )
        valid, reason = validate_archive(zip_path)
        assert valid is True
        assert reason is None

    def test_archive_exceeding_max_files_rejected(self, tmp_path):
        entries = {f"file{i}.txt": b"x" for i in range(51)}  # limit is 50
        zip_path = _make_zip(tmp_path / "too_many.zip", entries)
        valid, reason = validate_archive(zip_path)
        assert valid is False
        assert reason == UploadRejectionReason.ARCHIVE_TOO_MANY_FILES

    def test_archive_exceeding_extracted_size_rejected(self, tmp_path):
        # Limit is 104_857_600 bytes (100 MB); create entries that report
        # a large uncompressed size using highly compressible content.
        big_content = b"0" * (60 * 1024 * 1024)  # 60 MB per file, compresses well
        zip_path = _make_zip(
            tmp_path / "too_big.zip",
            {"big1.bin": big_content, "big2.bin": big_content},
        )
        valid, reason = validate_archive(zip_path)
        assert valid is False
        assert reason == UploadRejectionReason.ARCHIVE_TOO_LARGE

    def test_archive_nested_zip_beyond_depth_rejected(self, tmp_path):
        # max_depth is 2; place a nested zip at depth >= 2 (2 slashes)
        zip_path = _make_zip(
            tmp_path / "nested.zip",
            {"a/b/inner.zip": b"fake zip bytes"},
        )
        valid, reason = validate_archive(zip_path)
        assert valid is False
        assert reason == UploadRejectionReason.ARCHIVE_TOO_DEEP

    def test_archive_nested_zip_within_depth_allowed(self, tmp_path):
        # depth 0 (no directory) nested zip should be allowed
        zip_path = _make_zip(
            tmp_path / "shallow_nested.zip",
            {"inner.zip": b"fake zip bytes"},
        )
        valid, reason = validate_archive(zip_path)
        assert valid is True
        assert reason is None


# --- get_quarantine_path ---


class TestGetQuarantinePath:
    def test_creates_directory_if_missing(self, tmp_path):
        quarantine_dir = tmp_path / "quarantine_test"
        with patch.object(
            upload_config.settings, "upload_quarantine_path", str(quarantine_dir)
        ):
            result = get_quarantine_path()
            assert result.exists()
            assert result.is_dir()

    def test_returns_resolved_absolute_path(self, tmp_path):
        quarantine_dir = tmp_path / "quarantine_resolve"
        with patch.object(
            upload_config.settings, "upload_quarantine_path", str(quarantine_dir)
        ):
            result = get_quarantine_path()
            assert result.is_absolute()

    def test_idempotent_when_directory_exists(self, tmp_path):
        quarantine_dir = tmp_path / "quarantine_existing"
        quarantine_dir.mkdir()
        with patch.object(
            upload_config.settings, "upload_quarantine_path", str(quarantine_dir)
        ):
            result = get_quarantine_path()
            assert result.exists()

    @pytest.mark.skipif(
        __import__("platform").system() == "Windows",
        reason="POSIX file permissions are not enforced the same way on Windows",
    )
    def test_directory_has_restricted_permissions(self, tmp_path):
        import stat

        quarantine_dir = tmp_path / "quarantine_perms"
        with patch.object(
            upload_config.settings, "upload_quarantine_path", str(quarantine_dir)
        ):
            result = get_quarantine_path()
            mode = stat.S_IMODE(result.stat().st_mode)
            assert mode == 0o700


# --- sanitize_filename ---


class TestSanitizeFilename:
    def test_removes_path_traversal_sequences(self):
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_removes_backslash_path_traversal(self):
        result = sanitize_filename("..\\..\\windows\\system32")
        assert ".." not in result
        assert "\\" not in result

    def test_removes_null_bytes(self):
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_removes_control_characters(self):
        result = sanitize_filename("file\x01\x02\x03name.txt")
        assert all(ord(ch) >= 0x20 for ch in result)

    def test_collapses_multiple_dots(self):
        result = sanitize_filename("file....txt")
        assert "..." not in result

    def test_limits_length_to_255_chars(self):
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_normal_filename_unchanged(self):
        result = sanitize_filename("report_2024.pdf")
        assert result == "report_2024.pdf"


# --- Rejection rate tracking ---


class TestRejectionRateLimiting:
    def test_user_under_limit_is_not_rate_limited(self):
        user_id = "user-under-limit"
        for _ in range(5):  # limit is 10
            record_rejection(user_id)
        assert is_rate_limited(user_id) is False

    def test_user_at_limit_is_rate_limited(self):
        user_id = "user-at-limit"
        for _ in range(10):  # exactly at limit
            record_rejection(user_id)
        assert is_rate_limited(user_id) is True

    def test_user_over_limit_is_rate_limited(self):
        user_id = "user-over-limit"
        for _ in range(15):
            record_rejection(user_id)
        assert is_rate_limited(user_id) is True

    def test_user_with_no_rejections_is_not_rate_limited(self):
        assert is_rate_limited("brand-new-user") is False

    def test_old_rejections_outside_window_are_not_counted(self):
        user_id = "user-old-rejections"
        old_time = datetime.utcnow() - timedelta(minutes=120)  # window is 60 min

        with patch("app.services.upload_config.datetime") as mock_dt:
            mock_dt.utcnow.return_value = old_time
            for _ in range(15):
                record_rejection(user_id)

        # Now check with real "current" time - old entries should be filtered out
        assert is_rate_limited(user_id) is False

    def test_cleanup_expired_entries_removes_stale_users(self):
        user_id = "user-to-cleanup"
        old_time = datetime.utcnow() - timedelta(minutes=120)

        with patch("app.services.upload_config.datetime") as mock_dt:
            mock_dt.utcnow.return_value = old_time
            record_rejection(user_id)

        cleaned = cleanup_expired_entries()
        assert cleaned >= 1
        assert user_id not in upload_config._rejection_tracker

    def test_cleanup_keeps_recent_entries(self):
        user_id = "user-recent"
        record_rejection(user_id)
        cleanup_expired_entries()
        assert user_id in upload_config._rejection_tracker
