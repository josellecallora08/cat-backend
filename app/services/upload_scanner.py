"""Malware scanning interface for uploaded files.

Connects to ClamAV daemon to scan files before they are accepted into the system.

**Fail-closed behavior**: If the malware scanner is unreachable or encounters any
error, files are REJECTED (not accepted). This ensures that a scanner outage does
not silently allow malicious files through the upload pipeline.
"""

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of a malware scan on an uploaded file.

    Attributes:
        clean: True if the file passed scanning (no malware detected).
        signature: Name of the malware signature if detected (e.g. "Eicar-Signature").
        error: Error message if the scanner could not complete the scan.
               When error is set, clean is always False (fail-closed).
    """

    clean: bool
    signature: Optional[str] = None
    error: Optional[str] = None


def scan_file(file_path: Path) -> ScanResult:
    """Scan a file for malware using ClamAV daemon.

    Connects to ClamAV via Unix socket (or network socket as fallback on Windows
    or when the socket file is not found).

    Fail-closed behavior:
        - If scanning is disabled via settings, returns clean=True (bypass).
        - If the scanner is unreachable, returns clean=False with an error message.
        - If any unexpected exception occurs, returns clean=False with the error.

    Args:
        file_path: Path to the file to scan.

    Returns:
        ScanResult indicating whether the file is clean, infected, or if an error
        occurred during scanning.
    """
    if not settings.upload_scanner_enabled:
        logger.debug("Malware scanning disabled, skipping scan for %s", file_path)
        return ScanResult(clean=True)

    try:
        import clamd  # noqa: F401 - optional dependency
    except ImportError:
        logger.error("clamd library not installed, cannot perform malware scan")
        return ScanResult(clean=False, error="Scanner unavailable")

    try:
        cd = _connect_to_clamd()
    except Exception as exc:
        logger.error("Failed to connect to ClamAV daemon: %s", exc)
        return ScanResult(clean=False, error="Scanner unavailable")

    try:
        result = cd.scan(str(file_path))
    except Exception as exc:
        logger.error("ClamAV scan failed for %s: %s", file_path, exc)
        return ScanResult(clean=False, error="Scanner unavailable")

    return _parse_scan_result(result, file_path)


def _connect_to_clamd():
    """Establish connection to ClamAV daemon.

    Attempts Unix socket first, falls back to network socket on Windows
    or if the socket file doesn't exist.

    Returns:
        A connected clamd client instance.

    Raises:
        Exception: If connection cannot be established via any method.
    """
    import clamd

    socket_path = settings.upload_scanner_socket

    # On Windows or if the Unix socket doesn't exist, use network socket
    if platform.system() == "Windows" or not Path(socket_path).exists():
        logger.debug("Using ClamAV network socket (TCP) connection")
        cd = clamd.ClamdNetworkSocket()
        cd.ping()
        return cd

    # Prefer Unix socket connection
    logger.debug("Using ClamAV Unix socket at %s", socket_path)
    try:
        cd = clamd.ClamdUnixSocket(path=socket_path)
        cd.ping()
        return cd
    except (ConnectionError, OSError) as exc:
        # Fallback to network socket if Unix socket fails
        logger.warning(
            "Unix socket connection failed (%s), trying network socket", exc
        )
        cd = clamd.ClamdNetworkSocket()
        cd.ping()
        return cd


def _parse_scan_result(result: dict, file_path: Path) -> ScanResult:
    """Parse the raw ClamAV scan result dict into a ScanResult.

    ClamAV returns a dict like:
        {'/path/to/file': ('OK', None)} for clean files
        {'/path/to/file': ('FOUND', 'Eicar-Signature')} for infected files

    Args:
        result: Raw result dict from clamd.scan().
        file_path: The scanned file path (used for dict lookup).

    Returns:
        Parsed ScanResult.
    """
    if result is None:
        # No result means file was not found or scan didn't complete
        logger.error("ClamAV returned no result for %s", file_path)
        return ScanResult(clean=False, error="Scanner unavailable")

    file_key = str(file_path)
    file_result = result.get(file_key)

    if file_result is None:
        # Try to get any result (ClamAV may use absolute path)
        for key, value in result.items():
            file_result = value
            break

    if file_result is None:
        logger.error("Could not parse ClamAV result for %s: %s", file_path, result)
        return ScanResult(clean=False, error="Scanner unavailable")

    status = file_result[0]
    signature = file_result[1] if len(file_result) > 1 else None

    if status == "OK":
        logger.info("File %s passed malware scan", file_path)
        return ScanResult(clean=True)
    elif status == "FOUND":
        logger.warning(
            "Malware detected in %s: signature=%s", file_path, signature
        )
        return ScanResult(clean=False, signature=signature)
    else:
        # Unexpected status (e.g. "ERROR") — fail-closed
        logger.error(
            "Unexpected ClamAV status for %s: status=%s, detail=%s",
            file_path,
            status,
            signature,
        )
        return ScanResult(clean=False, error=f"Unexpected scanner status: {status}")
