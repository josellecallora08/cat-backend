"""Persistence tests for negotiation standards and immutable versions."""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as OrmSession

from app.database import Base
from app.models import (
    Campaign,
    Evaluation,
    ImmutableVersionError,
    NegotiationStandard,
    NegotiationStandardVersion,
    Session,
    User,
)


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with OrmSession(engine) as session:
        yield session
    engine.dispose()


def _user(role: str = "admin") -> User:
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.test",
        full_name="Admin User",
        role=role,
        hashed_password="not-a-password",
        is_active=True,
    )


def _campaign() -> Campaign:
    return Campaign(id=uuid.uuid4(), name=f"Campaign {uuid.uuid4()}")


def test_standard_defaults_and_campaign_uniqueness(sqlite_session: OrmSession) -> None:
    admin = _user()
    campaign = _campaign()
    sqlite_session.add_all([admin, campaign])
    sqlite_session.flush()

    standard = NegotiationStandard(
        campaign_id=campaign.id,
        name="Default standard",
        created_by=admin.id,
        updated_by=admin.id,
    )
    sqlite_session.add(standard)
    sqlite_session.flush()

    assert standard.status == "draft"
    assert standard.revision == 1

    duplicate = NegotiationStandard(
        campaign_id=campaign.id,
        name="Duplicate",
        created_by=admin.id,
        updated_by=admin.id,
    )
    sqlite_session.add(duplicate)
    with pytest.raises(IntegrityError):
        sqlite_session.flush()


def test_version_snapshot_round_trips_and_is_unique(sqlite_session: OrmSession) -> None:
    admin = _user()
    campaign = _campaign()
    standard = NegotiationStandard(
        campaign=campaign,
        name="Standard",
        created_by=admin.id,
        updated_by=admin.id,
    )
    sqlite_session.add_all([admin, standard])
    sqlite_session.flush()

    version = NegotiationStandardVersion(
        standard_id=standard.id,
        version_number=1,
        snapshot={"schema_version": 1, "blocks": []},
        content_hash="a" * 64,
        created_by=admin.id,
        published_by=admin.id,
    )
    sqlite_session.add(version)
    sqlite_session.flush()

    assert version.snapshot == {"schema_version": 1, "blocks": []}
    assert standard.versions[0].version_number == 1


def test_version_update_and_delete_are_blocked(sqlite_session: OrmSession) -> None:
    admin = _user()
    campaign = _campaign()
    standard = NegotiationStandard(
        campaign=campaign,
        name="Standard",
        created_by=admin.id,
        updated_by=admin.id,
    )
    version = NegotiationStandardVersion(
        standard=standard,
        version_number=1,
        snapshot={"blocks": []},
        content_hash="b" * 64,
        created_by=admin.id,
        published_by=admin.id,
    )
    sqlite_session.add_all([admin, standard, version])
    sqlite_session.flush()

    version.publication_note = "changed"
    with pytest.raises(ImmutableVersionError):
        sqlite_session.flush()

    sqlite_session.rollback()
    replacement = NegotiationStandardVersion(
        standard_id=standard.id,
        version_number=1,
        snapshot={"blocks": []},
        content_hash="b" * 64,
        created_by=admin.id,
        published_by=admin.id,
    )
    sqlite_session.add(replacement)
    sqlite_session.flush()
    sqlite_session.delete(replacement)
    with pytest.raises(ImmutableVersionError):
        sqlite_session.flush()
