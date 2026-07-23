"""Unit tests for app.services.upload_validator."""

import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.upload_validator import (
    UploadRejectionReason,
    validate_extension,
    validate_mime_type,
    validate_file_signature,
    validate_file_size_streaming,
    validate_docx_archive,
)


class TestValidateExtension:
    """Task 13.1 + 13.2: Test accepted and rejected extensions."""

    @pytest.mark.parametrize("filename", [
        "doc.pdf", "report.docx", "notes.txt", "data.csv", "readme.md",
        "DOC.PDF", "Report.DOCX", "NOTES.TXT", "DATA.CSV", "README.MD",
    ])
    def test_accepted_formats(self, filename):
        valid, reason = validate_extension(filename)
        assert valid is True
        assert reason is None

    @pytest.mark.parametrize("filename", [
        "sheet.xlsx", "audio.mp3", "sound.wav", "archive.zip",
        "program.exe", "image.png", "script.sh", "noext",
    ])
    def test_rejected_formats(self, filename):
        valid, reason = validate_extension(filename)
        assert valid is False
        assert reason == UploadRejectionReason.INVALID_EXTENSION


class TestValidateMimeType:
    """Task 13.3: Test MIME type mismatch detection."""

    @pytest.mark.parametrize("filename,content_type", [
        ("doc.pdf", "application/pdf"),
        ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("notes.txt", "text/plain"),
        ("data.csv", "text/csv"),
        ("readme.md", "text/markdown"),
        ("readme.md", "text/x-markdown"),
    ])
    def test_valid_mime_types(self, filename, content_type):
        valid, reason = validate_mime_type(filename, content_type)
        assert valid is True
        assert reason is None

    @pytest.mark.parametrize("filename,content_type", [
        ("doc.pdf", "application/octet-stream"),
        ("report.docx", "text/plain"),
        ("notes.txt", "application/pdf"),
        ("data.csv", "audio/mpeg"),
        ("readme.md", "application/zip"),
    ])
    def test_mime_mismatch(self, filename, content_type):
        valid, reason = validate_mime_type(filename, content_type)
        assert valid is False
        assert reason == UploadRejectionReason.MIME_MISMATCH


class TestValidateFileSignature:
    """Task 13.4: Test binary signature validation."""

    def test_valid_pdf_signature(self):
        header = b"\x25\x50\x44\x46\x2d\x31\x2e\x34"  # %PDF-1.4
        valid, reason = validate_file_signature("doc.pdf", header)
        assert valid is True

    def test_valid_docx_signature(self):
        header = b"\x50\x4B\x03\x04\x00\x00\x00\x00"  # PK..
        valid, reason = validate_file_signature("report.docx", header)
        assert valid is True

    def test_valid_txt_signature(self):
        header = b"Hello, this is plain text content"
        valid, reason = validate_file_signature("notes.txt", header)
        assert valid is True

    def test_valid_csv_signature(self):
        header = b"name,age,city\n"
        valid, reason = validate_file_signature("data.csv", header)
        assert valid is True

    def test_valid_md_signature(self):
        header = b"# Heading\n\nSome markdown"
        valid, reason = validate_file_signature("readme.md", header)
        assert valid is True

    def test_pdf_with_exe_signature_rejected(self):
        header = b"\x4D\x5A\x90\x00\x03\x00\x00\x00"  # MZ (EXE)
        valid, reason = validate_file_signature("fake.pdf", header)
        assert valid is False
        assert reason == UploadRejectionReason.SIGNATURE_MISMATCH

    def test_txt_with_pdf_signature_rejected(self):
        header = b"\x25\x50\x44\x46\x2d\x31\x2e\x34"  # %PDF (binary sig in text file)
        valid, reason = validate_file_signature("fake.txt", header)
        assert valid is False
        assert reason == UploadRejectionReason.SIGNATURE_MISMATCH

    def test_txt_with_binary_content_rejected(self):
        # Invalid UTF-8 bytes
        header = b"\xff\xfe\x00\x01\x80\x81\x82\x83"
        valid, reason = validate_file_signature("binary.txt", header)
        assert valid is False
        assert reason == UploadRejectionReason.SIGNATURE_MISMATCH


class TestValidateFileSizeStreaming:
    """Task 13.5: Test streaming size enforcement."""

    @pytest.fixture
    def make_upload_file(self):
        """Create a mock upload file with given content."""
        def _make(content: bytes):
            mock = AsyncMock()
            chunks = [content[i:i + 65536] for i in range(0, len(content), 65536)]
            chunks.append(b"")  # EOF
            mock.read = AsyncMock(side_effect=chunks)
            return mock
        return _make

    @pytest.mark.asyncio
    async def test_file_under_limit(self, make_upload_file):
        content = b"x" * 1000
        file = make_upload_file(content)
        result_bytes, reason = await validate_file_size_streaming(file, 10_485_760)
        assert reason is None
        assert result_bytes == content

    @pytest.mark.asyncio
    async def test_file_exactly_at_limit(self, make_upload_file):
        content = b"x" * 10_485_760
        file = make_upload_file(content)
        result_bytes, reason = await validate_file_size_streaming(file, 10_485_760)
        assert reason is None
        assert len(result_bytes) == 10_485_760

    @pytest.mark.asyncio
    async def test_file_over_limit(self, make_upload_file):
        content = b"x" * 10_485_761
        file = make_upload_file(content)
        result_bytes, reason = await validate_file_size_streaming(file, 10_485_760)
        assert reason == UploadRejectionReason.FILE_TOO_LARGE


class TestValidateDocxArchive:
    """Task 13.6: Test DOCX archive safety limits."""

    def _make_docx_zip(self, tmp_path, entries: dict) -> Path:
        """Create a minimal DOCX-like ZIP."""
        path = tmp_path / "test.docx"
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in entries.items():
                zf.writestr(name, content)
        return path

    def test_valid_docx_within_limits(self, tmp_path):
        entries = {"word/document.xml": "<w:document/>", "[Content_Types].xml": "<Types/>"}
        path = self._make_docx_zip(tmp_path, entries)
        valid, reason = validate_docx_archive(path)
        assert valid is True
        assert reason is None

    def test_docx_too_many_entries(self, tmp_path):
        entries = {f"file{i}.xml": "x" for i in range(501)}  # default limit is 500
        path = self._make_docx_zip(tmp_path, entries)
        valid, reason = validate_docx_archive(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_TOO_MANY_ENTRIES

    def test_docx_too_large_uncompressed(self, tmp_path):
        # 50 MB limit; create entries exceeding it
        big = b"0" * (30 * 1024 * 1024)  # 30 MB each
        path = tmp_path / "big.docx"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("big1.xml", big)
            zf.writestr("big2.xml", big)
        valid, reason = validate_docx_archive(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_TOO_LARGE_UNCOMPRESSED

    def test_docx_too_deep(self, tmp_path):
        # depth limit is 2; entry with 2+ slashes should fail
        entries = {"a/b/deep.xml": "<nested/>"}
        path = self._make_docx_zip(tmp_path, entries)
        valid, reason = validate_docx_archive(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_TOO_DEEP
