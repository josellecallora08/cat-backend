"""Property-based tests for S1-11 quarantine cleanup correctness."""

import logging
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.services import upload_quarantine
from app.services.upload_quarantine import CleanupResult, cleanup_expired_files


PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _session_factory(upload=None, *, commit_error=None, query_error=None):
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = upload
    session.execute = AsyncMock(
        side_effect=query_error if query_error is not None else None,
        return_value=result,
    )
    session.commit = AsyncMock(side_effect=commit_error)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=context)


def _expire(path: Path, age_seconds: float) -> None:
    modified = time.time() - age_seconds
    os.utime(path, (modified, modified))


@PROPERTY_SETTINGS
@given(
    retention=st.integers(min_value=0, max_value=8760),
    age_delta=st.one_of(
        st.integers(min_value=-10_000, max_value=-5),
        st.integers(min_value=5, max_value=10_000),
    ),
)
async def test_property_expiration_correctness(tmp_path, monkeypatch, retention, age_delta):
    """Property 1: only files beyond retention remain eligible (zero means all)."""
    source = tmp_path / f"{uuid.uuid4()}.pdf"
    source.write_bytes(b"raw")
    _expire(source, retention * 3600 + age_delta)
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", retention)

    await cleanup_expired_files(_session_factory())

    should_delete = retention == 0 or age_delta > 0
    assert source.exists() is not should_delete
    source.unlink(missing_ok=True)


@PROPERTY_SETTINGS
@given(
    retention=st.integers(min_value=0, max_value=8760),
    age_seconds=st.integers(min_value=-10_000, max_value=40_000_000),
)
async def test_property_temp_file_exclusion(tmp_path, monkeypatch, retention, age_seconds):
    """Property 5: in-progress uploads survive regardless of age or retention."""
    source = tmp_path / f".tmp_upload_{uuid.uuid4()}"
    source.write_bytes(b"partial")
    _expire(source, age_seconds)
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", retention)

    result = await cleanup_expired_files(_session_factory())

    assert source.exists()
    assert result == CleanupResult()
    source.unlink()


@PROPERTY_SETTINGS
@given(
    content_hash=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
    extracted_content=st.text(max_size=200),
    scan_status=st.sampled_from(["pending", "clean", "infected", "error"]),
    filename=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
        min_size=1,
        max_size=30,
    ),
)
async def test_property_metadata_preservation(
    tmp_path,
    monkeypatch,
    content_hash,
    extracted_content,
    scan_status,
    filename,
):
    """Property 2: cleanup does not mutate compliance metadata."""
    storage_key = f"{uuid.uuid4()}.txt"
    source = tmp_path / storage_key
    source.write_bytes(b"raw")
    upload = SimpleNamespace(
        id=uuid.uuid4(),
        deleted_at=None,
        content_hash=content_hash,
        extracted_content=extracted_content,
        scan_status=scan_status,
        scan_signature="signature",
        uploaded_by=uuid.uuid4(),
        script_id=uuid.uuid4(),
        filename_original=filename,
        mime_type="text/plain",
        file_size_bytes=3,
        rejection_reason="retained",
    )
    preserved = {key: value for key, value in vars(upload).items() if key not in {"deleted_at"}}
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", 0)

    result = await cleanup_expired_files(_session_factory(upload))

    assert result == CleanupResult(deleted=1)
    assert {key: getattr(upload, key) for key in preserved} == preserved


@PROPERTY_SETTINGS
@given(reason=st.text(min_size=1, max_size=80))
async def test_property_database_failure_safety(tmp_path, monkeypatch, reason):
    """Property 3: any database failure preserves the corresponding file."""
    storage_key = f"{uuid.uuid4()}.pdf"
    source = tmp_path / storage_key
    source.write_bytes(b"raw")
    upload = SimpleNamespace(id=uuid.uuid4(), deleted_at=None)
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", 0)

    result = await cleanup_expired_files(
        _session_factory(upload, commit_error=RuntimeError(reason))
    )

    assert source.exists()
    assert result == CleanupResult(skipped=1, errors=1)
    source.unlink()


@PROPERTY_SETTINGS
@given(extension=st.sampled_from([".pdf", ".docx", ".txt", ".csv", ".md"]))
async def test_property_orphan_file_handling(tmp_path, monkeypatch, caplog, extension):
    """Property 4: expired orphan files are removed with structured warnings."""
    caplog.clear()
    storage_key = f"{uuid.uuid4()}{extension}"
    source = tmp_path / storage_key
    source.write_bytes(b"raw")
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", 0)

    with caplog.at_level(logging.WARNING, logger=upload_quarantine.__name__):
        result = await cleanup_expired_files(_session_factory())

    assert result == CleanupResult(orphaned=1)
    assert not source.exists()
    orphan_logs = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "orphaned_quarantine_file_removed"
    ]
    assert any(record.storage_key == storage_key for record in orphan_logs)


@PROPERTY_SETTINGS
@given(removable_count=st.integers(min_value=1, max_value=5))
async def test_property_filesystem_error_resilience(tmp_path, monkeypatch, removable_count):
    """Property 6: one deletion error does not prevent other removals."""
    blocked = tmp_path / f"blocked-{uuid.uuid4()}.pdf"
    blocked.write_bytes(b"blocked")
    removable = []
    for _ in range(removable_count):
        source = tmp_path / f"{uuid.uuid4()}.pdf"
        source.write_bytes(b"raw")
        removable.append(source)
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", 0)
    real_unlink = Path.unlink

    def selective_unlink(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("denied")
        return real_unlink(path, *args, **kwargs)

    with patch.object(Path, "unlink", selective_unlink):
        result = await cleanup_expired_files(_session_factory())

    assert blocked.exists()
    assert all(not source.exists() for source in removable)
    assert result == CleanupResult(orphaned=removable_count, skipped=1, errors=1)
    blocked.unlink()


@PROPERTY_SETTINGS
@given(
    removed_count=st.integers(min_value=0, max_value=4),
    failed_count=st.integers(min_value=0, max_value=4),
)
async def test_property_summary_logging_accuracy(
    tmp_path, monkeypatch, caplog, removed_count, failed_count
):
    """Property 7: structured summary counts exactly match actual outcomes."""
    caplog.clear()
    removable = []
    failed = []
    for _ in range(removed_count):
        source = tmp_path / f"{uuid.uuid4()}.pdf"
        source.write_bytes(b"raw")
        removable.append(source)
    for _ in range(failed_count):
        source = tmp_path / f"failed-{uuid.uuid4()}.pdf"
        source.write_bytes(b"raw")
        failed.append(source)
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_path", str(tmp_path))
    monkeypatch.setattr(upload_quarantine.settings, "upload_quarantine_retention_hours", 0)
    real_unlink = Path.unlink

    def selective_unlink(path, *args, **kwargs):
        if path in failed:
            raise OSError("simulated failure")
        return real_unlink(path, *args, **kwargs)

    with (
        caplog.at_level(logging.INFO, logger=upload_quarantine.__name__),
        patch.object(Path, "unlink", selective_unlink),
    ):
        result = await cleanup_expired_files(_session_factory())

    assert result == CleanupResult(
        orphaned=removed_count,
        skipped=failed_count,
        errors=failed_count,
    )
    summaries = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "quarantine_cleanup_completed"
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.deleted == 0
    assert summary.orphaned == removed_count
    assert summary.skipped == failed_count
    assert summary.errors == failed_count
    assert summary.total_removed == removed_count
    for source in failed:
        source.unlink()
