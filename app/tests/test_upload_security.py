"""Security tests for upload validation: spoofing, zip-bomb, traversal, and malware."""

import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.upload_validator import (
    UploadRejectionReason,
    validate_extension,
    validate_file_signature,
    validate_docx_archive,
    validate_mime_type,
)
from app.services.upload_quarantine import sanitize_filename
from app.services.upload_scanner import ScanResult, scan_file
from app.services import upload_scanner


class TestExtensionSpoofing:
    """Task 17.1: Renamed .exe to .pdf with correct MIME but wrong magic bytes."""

    def test_exe_renamed_to_pdf_rejected(self):
        """Attacker renames .exe to .pdf and sets correct MIME, but magic bytes are MZ."""
        # Extension check passes
        valid, _ = validate_extension("malware.pdf")
        assert valid is True

        # MIME check passes (attacker controls Content-Type header)
        valid, _ = validate_mime_type("malware.pdf", "application/pdf")
        assert valid is True

        # But binary signature check catches it!
        exe_header = b"\x4D\x5A\x90\x00\x03\x00\x00\x00"  # MZ header
        valid, reason = validate_file_signature("malware.pdf", exe_header)
        assert valid is False
        assert reason == UploadRejectionReason.SIGNATURE_MISMATCH

    def test_zip_renamed_to_txt_rejected(self):
        """Attacker renames .zip to .txt, claims text/plain MIME."""
        valid, _ = validate_extension("payload.txt")
        assert valid is True

        valid, _ = validate_mime_type("payload.txt", "text/plain")
        assert valid is True

        # PK signature in a .txt file triggers binary content detection
        zip_header = b"\x50\x4B\x03\x04\x00\x00\x00\x00"
        valid, reason = validate_file_signature("payload.txt", zip_header)
        assert valid is False
        assert reason == UploadRejectionReason.SIGNATURE_MISMATCH

    def test_non_utf8_in_csv_rejected(self):
        """Attacker puts raw binary in a .csv file."""
        binary_content = b"\x89PNG\r\n\x1a\n"  # PNG header
        valid, reason = validate_file_signature("data.csv", binary_content)
        assert valid is False
        assert reason == UploadRejectionReason.SIGNATURE_MISMATCH


class TestDocxZipBomb:
    """Task 17.2: DOCX zip bomb with huge uncompressed size."""

    def test_docx_zip_bomb_rejected(self, tmp_path):
        """DOCX claiming small compressed but huge uncompressed size."""
        path = tmp_path / "bomb.docx"
        # Create a DOCX-like ZIP with highly compressible content
        # Total uncompressed > 50 MB (settings.upload_max_docx_uncompressed_size)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Each entry is 30 MB of zeros (compresses very small)
            zf.writestr("content1.xml", b"0" * (30 * 1024 * 1024))
            zf.writestr("content2.xml", b"0" * (30 * 1024 * 1024))

        valid, reason = validate_docx_archive(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_TOO_LARGE_UNCOMPRESSED

    def test_docx_many_entries_rejected(self, tmp_path):
        """DOCX with excessive number of entries (zip bomb variant)."""
        path = tmp_path / "many_entries.docx"
        with zipfile.ZipFile(path, "w") as zf:
            for i in range(501):  # Limit is 500
                zf.writestr(f"entry_{i}.xml", "x")

        valid, reason = validate_docx_archive(path)
        assert valid is False
        assert reason == UploadRejectionReason.DOCX_TOO_MANY_ENTRIES


class TestPathTraversal:
    """Task 17.3: Path traversal in filename."""

    @pytest.mark.parametrize("malicious_name,must_not_contain", [
        ("../../etc/passwd", "/"),
        ("..\\..\\windows\\system32\\config\\sam", "\\"),
        ("../../../root/.ssh/id_rsa", "/"),
        ("%2e%2e/etc/shadow", "/"),  # URL-encoded traversal (/ stripped)
    ])
    def test_path_traversal_sanitized(self, malicious_name, must_not_contain):
        result = sanitize_filename(malicious_name)
        assert must_not_contain not in result
        assert ".." not in result

    def test_null_byte_injection(self):
        """Null byte injection to truncate filename."""
        result = sanitize_filename("legit.pdf\x00.exe")
        assert "\x00" not in result

    def test_control_character_injection(self):
        """Control characters removed."""
        result = sanitize_filename("file\x01\x02\x03\x1fname.pdf")
        assert all(ord(c) >= 0x20 for c in result)


class TestMalwareDetection:
    """Task 17.4: EICAR test string triggers malware rejection."""

    def test_eicar_triggers_rejection_when_scanner_available(self, tmp_path):
        """When ClamAV is running, EICAR test string should be detected."""
        # EICAR standard test string
        eicar = (
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-"
            b"STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        )
        test_file = tmp_path / "eicar.txt"
        test_file.write_bytes(eicar)

        # Create a mock clamd module so the import inside scan_file succeeds
        mock_clamd_module = MagicMock()
        with patch.dict(sys.modules, {"clamd": mock_clamd_module}):
            # Mock ClamAV detecting the EICAR signature
            with patch("app.services.upload_scanner._connect_to_clamd") as mock_connect:
                mock_cd = MagicMock()
                mock_cd.scan.return_value = {
                    str(test_file): ("FOUND", "Eicar-Signature")
                }
                mock_connect.return_value = mock_cd

                with patch.object(
                    upload_scanner.settings,
                    "upload_scanner_enabled",
                    True,
                ):
                    result = scan_file(test_file)
                    assert result.clean is False
                    assert result.signature == "Eicar-Signature"

    def test_scanner_unavailable_fails_closed(self, tmp_path):
        """When scanner cannot connect, upload is rejected (fail-closed)."""
        test_file = tmp_path / "file.pdf"
        test_file.write_bytes(b"%PDF-1.4 content")

        # Create a mock clamd module so the import inside scan_file succeeds
        mock_clamd_module = MagicMock()
        with patch.dict(sys.modules, {"clamd": mock_clamd_module}):
            with patch("app.services.upload_scanner._connect_to_clamd") as mock_connect:
                mock_connect.side_effect = ConnectionError("Cannot connect")

                with patch.object(
                    upload_scanner.settings,
                    "upload_scanner_enabled",
                    True,
                ):
                    result = scan_file(test_file)
                    assert result.clean is False
                    assert result.error == "Scanner unavailable"
