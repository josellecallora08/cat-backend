"""Database integration coverage for quarantine cleanup metadata preservation."""

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.models  # noqa: F401 - register all mapped tables
from app.database import Base
from app.models.script_upload import ScriptUpload
from app.services import upload_quarantine
from app.services.upload_quarantine import CleanupResult, cleanup_expired_files


async def test_cleanup_changes_only_deleted_at_in_database(tmp_path, monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    storage_key = f"{uuid.uuid4()}.txt"
    source = tmp_path / storage_key
    source.write_bytes(b"raw source")
    expired_time = time.time() - 3600
    os.utime(source, (expired_time, expired_time))

    original_updated_at = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    upload_id = uuid.uuid4()
    script_id = uuid.uuid4()
    uploader_id = uuid.uuid4()
    rejected_by = uuid.uuid4()
    rejected_at = datetime(2020, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

    upload = ScriptUpload(
        id=upload_id,
        filename_original="original.txt",
        mime_type="text/plain",
        file_size_bytes=10,
        content_hash="a" * 64,
        storage_key=storage_key,
        uploaded_by=uploader_id,
        scan_status="clean",
        scan_signature="known-clean",
        extraction_status="completed",
        extracted_content="sanitized content",
        status="completed",
        script_id=script_id,
        rejected_at=rejected_at,
        rejected_by=rejected_by,
        rejection_reason="retained reason",
        quarantine_expires_at=expires_at,
        updated_at=original_updated_at,
    )
    async with session_factory() as session:
        session.add(upload)
        await session.commit()
        await session.refresh(upload)
        preserved_before = {
            "content_hash": upload.content_hash,
            "extracted_content": upload.extracted_content,
            "scan_status": upload.scan_status,
            "scan_signature": upload.scan_signature,
            "uploaded_by": upload.uploaded_by,
            "script_id": upload.script_id,
            "filename_original": upload.filename_original,
            "mime_type": upload.mime_type,
            "file_size_bytes": upload.file_size_bytes,
            "created_at": upload.created_at,
            "updated_at": upload.updated_at,
            "quarantine_expires_at": upload.quarantine_expires_at,
            "rejected_at": upload.rejected_at,
            "rejected_by": upload.rejected_by,
            "rejection_reason": upload.rejection_reason,
        }

    monkeypatch.setattr(
        upload_quarantine.settings, "upload_quarantine_path", str(tmp_path)
    )
    monkeypatch.setattr(
        upload_quarantine.settings, "upload_quarantine_retention_hours", 0
    )

    result = await cleanup_expired_files(session_factory)

    assert result == CleanupResult(deleted=1)
    assert not source.exists()
    async with session_factory() as session:
        persisted = await session.get(ScriptUpload, upload_id)
        assert persisted.deleted_at is not None
        preserved_after = {
            field: getattr(persisted, field) for field in preserved_before
        }
        assert preserved_after == preserved_before

    await engine.dispose()
