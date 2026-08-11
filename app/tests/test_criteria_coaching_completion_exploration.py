"""Focused bug-condition exploration for TASK-065 before application fixes."""

import subprocess
import sys


def test_session_api_import_and_startup_smoke():
    """The session router must import before endpoint tests are meaningful."""
    result = subprocess.run(
        [sys.executable, "-c", "import app.api.sessions; import app.main"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
