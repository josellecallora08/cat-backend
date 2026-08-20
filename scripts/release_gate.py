"""Run backend release checks and emit redacted, blocking gate results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from app.services.report_failures import FailureClass, redact_sensitive


@dataclass(frozen=True, slots=True)
class GateResult:
    """A stable, safe record for one release-gate check."""

    name: str
    failure_class: str
    component: str
    command: str
    passed: bool
    required: bool
    remediation: str
    output: str | None = None


CHECKS: tuple[tuple[str, FailureClass, tuple[str, ...], str], ...] = (
    ("lint", FailureClass.BACKEND, ("ruff", "check", "app/"), "Fix Ruff findings."),
    (
        "format",
        FailureClass.BACKEND,
        ("ruff", "format", "--check", "app/"),
        "Run Ruff format and commit the formatted files.",
    ),
    (
        "typecheck",
        FailureClass.BACKEND,
        ("mypy", "app/", "--ignore-missing-imports"),
        "Resolve mypy errors in backend code.",
    ),
    (
        "migrations",
        FailureClass.BACKEND,
        ("alembic", "check"),
        "Create or apply the required migration.",
    ),
    (
        "report-tests",
        FailureClass.BACKEND,
        ("pytest", "-q", "app/tests/test_report_*.py", "app/tests/test_sessions_api.py"),
        "Fix report contract and session test failures.",
    ),
    (
        "unit-tests",
        FailureClass.BACKEND,
        ("pytest", "-q", "-m", "not integration and not db_integration"),
        "Fix backend unit-test failures.",
    ),
    (
        "http-e2e",
        FailureClass.HTTP_E2E,
        ("pytest", "-q", "-m", "db_integration", "app/tests/test_report_http_e2e.py"),
        "Start the test database and fix the HTTP lifecycle assertion.",
    ),
    (
        "coverage",
        FailureClass.BACKEND,
        ("coverage", "report", "--fail-under=80"),
        "Add tests or reduce uncovered backend report code.",
    ),
)


def run_check(
    name: str, failure_class: FailureClass, command: tuple[str, ...], remediation: str
) -> GateResult:
    """Execute one check without invoking a shell and redact captured output."""
    # Commands are fixed internal tuples; shell execution remains disabled.
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = redact_sensitive((completed.stdout + "\n" + completed.stderr).strip())
    return GateResult(
        name=name,
        failure_class=failure_class.value,
        component="cat-backend",
        command=" ".join(command),
        passed=completed.returncode == 0,
        required=True,
        remediation=remediation,
        output=output[-2000:] if output else None,
    )


def write_artifact(path: Path, results: list[GateResult]) -> None:
    """Write a concise JSON artifact containing no raw command secrets."""
    payload = {
        "gate": "backend",
        "passed": all(result.passed for result in results if result.required),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run selected backend checks and block release success on required failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="append", choices=[check[0] for check in CHECKS])
    parser.add_argument("--artifact", type=Path, default=Path("release-gate-backend.json"))
    args = parser.parse_args()
    selected = set(args.check or [check[0] for check in CHECKS])
    results = [run_check(*check) for check in CHECKS if check[0] in selected]
    write_artifact(args.artifact, results)
    return 0 if all(result.passed for result in results if result.required) else 1


if __name__ == "__main__":
    sys.exit(main())
