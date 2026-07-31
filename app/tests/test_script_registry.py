"""Unit tests for the Script_Registry lifecycle service (task 7.11).

Feature: ai-debtor-script-contract

Complements the property-based tests in `test_property_script_lifecycle.py`
and `test_property_script_consumption.py` (tasks 7.2-7.10) with a concrete,
example-data-driven walk through the full Script_Registry lifecycle:

    create_draft -> publish -> update_draft (edit creates a new draft
    revision on a published script) -> publish again -> unpublish ->
    delete_script

The async in-memory SQLite fixture (`async_db`) and `User`/`Scenario` FK
prerequisite-row helpers below are adapted from the conventions established
in `test_property_script_lifecycle.py`.

_Requirements: 3.7, 3.9, 3.10_
"""

import json
import uuid

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Script, ScriptVersion, User
from app.models.script import ScriptStatus
from app.models.user import UserRole
from app.services.script_registry import (
    create_draft,
    delete_script,
    get_active_published_version,
    get_script,
    publish,
    unpublish,
    update_draft,
)

# --- Fixtures ---


@pytest.fixture
async def async_db():
    """Create an in-memory SQLite database for testing with foreign keys enabled."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    # Disable FK enforcement before dropping tables: Script.current_version_id
    # and ScriptVersion.script_id form a circular FK relationship, and once
    # rows populate both sides (e.g. a Published_Script), SQLite's
    # per-statement FK checking can otherwise reject DROP TABLE regardless of
    # drop order.
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys=OFF"))
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# --- FK prerequisite-row helpers ---


def _make_admin_user(name: str = "Test Admin") -> User:
    """Create a valid Administrator `User` instance for `created_by` FKs."""
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4()}@example.com",
        hashed_password="not-a-real-hash",
        full_name=name,
        role=UserRole.ADMIN.value,
        is_active=True,
    )


def _make_scenario(name: str = "Test Scenario") -> Scenario:
    """Create a valid `Scenario` instance for `scenario_id` FKs."""
    return Scenario(
        id=uuid.uuid4(),
        name=name,
        scenario_type="FINANCIAL_HARDSHIP",
        description="A test scenario for script lifecycle tests",
        debtor_profile={
            "name": "Maria Santos",
            "outstanding_balance": "5000.00",
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=True,
    )


# --- Concrete example Script_Contract content ---

CONTRACT_V1 = {
    "debtor_persona": {
        "name": "Maria Santos",
        "communication_style": "anxious and apologetic",
        "background": "single parent, recently lost a second job",
    },
    "financial_situation": {
        "outstanding_balance": 5000.00,
        "days_past_due": 45,
        "reason_for_delinquency": "temporary job loss",
    },
    "opening_response": "Hello, I'm calling about my account.",
    "expected_replies": [
        {"agent_statement": "Can you make a payment today?", "debtor_reply": "I can try to pay something small."}
    ],
    "trigger_phrases": [
        {"phrase": "legal action", "behavior": "become distressed"}
    ],
    "emotional_state_rules": [
        {"trigger": "threat", "state_change": "increase anxiety"}
    ],
    "payment_conditions": [
        {"condition": "partial payment", "term": "50 dollars now", "accepted": True}
    ],
    "escalation_conditions": [
        {"condition": "hostile language", "behavior": "end call", "ends_call": True}
    ],
    "prohibited_responses": ["I refuse to ever pay"],
    "conversation_goal": {
        "target_outcome": "payment plan agreed",
        "completion_condition": "debtor agrees to a plan",
    },
}

CONTRACT_V2 = {
    "debtor_persona": {
        "name": "Maria Santos",
        "communication_style": "calm but firm",
        "background": "single parent, recently found a new part-time job",
    },
    "financial_situation": {
        "outstanding_balance": 4200.00,
        "days_past_due": 60,
        "reason_for_delinquency": "medical expenses",
    },
    "opening_response": "Hi, I got your message about my balance.",
    "expected_replies": [
        {"agent_statement": "Can you make a payment today?", "debtor_reply": "I can pay part of it now."}
    ],
    "trigger_phrases": [
        {"phrase": "collections agency", "behavior": "become defensive"}
    ],
    "emotional_state_rules": [
        {"trigger": "threat", "state_change": "increase frustration"}
    ],
    "payment_conditions": [
        {"condition": "installment plan", "term": "100 dollars per month", "accepted": True}
    ],
    "escalation_conditions": [
        {"condition": "repeated threats", "behavior": "end call", "ends_call": True}
    ],
    "prohibited_responses": ["I will never pay this debt"],
    "conversation_goal": {
        "target_outcome": "installment plan agreed",
        "completion_condition": "debtor agrees to installments",
    },
}


# --- Tests ---


class TestScriptRegistryLifecycle:
    """Walks a single Script through its full lifecycle with concrete
    example data: create -> publish -> edit (new draft revision on a
    published script) -> publish again -> unpublish -> delete.

    _Requirements: 3.7, 3.9, 3.10_
    """

    async def test_full_lifecycle_sequence(self, async_db: AsyncSession):
        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        # 1. create_draft with a valid contract -> status="draft".
        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name="Lifecycle Test Script",
            scenario_id=scenario.id,
            format="json",
            raw_definition=json.dumps(CONTRACT_V1),
        )
        script_id = script.id
        assert script.status == ScriptStatus.DRAFT.value == "draft"

        # 2. publish -> ScriptVersion(version_number=1) created,
        # Script.status="published", current_version_id set to it.
        version_1 = await publish(async_db, script_id=script_id, admin_id=admin.id)
        assert version_1.version_number == 1
        assert version_1.script_id == script_id

        script_after_first_publish = await async_db.get(Script, script_id)
        assert script_after_first_publish is not None
        assert script_after_first_publish.status == ScriptStatus.PUBLISHED.value == "published"
        assert script_after_first_publish.current_version_id == version_1.id

        # 3. update_draft with different valid content (edit creates a new
        # draft revision, since the script is now published) -> draft_content
        # updated, status still "published", current_version_id unchanged
        # (still points at version 1).
        updated_script = await update_draft(
            async_db,
            script_id=script_id,
            raw_definition=json.dumps(CONTRACT_V2),
            format="json",
        )
        assert updated_script.draft_content == CONTRACT_V2
        assert updated_script.status == ScriptStatus.PUBLISHED.value == "published"
        assert updated_script.current_version_id == version_1.id

        # 4. publish again -> a SECOND ScriptVersion(version_number=2) is
        # created, current_version_id now points at version 2, and BOTH
        # ScriptVersion rows (1 and 2) still exist in the DB with their
        # original distinct content.
        version_2 = await publish(async_db, script_id=script_id, admin_id=admin.id)
        assert version_2.version_number == 2
        assert version_2.script_id == script_id
        assert version_2.id != version_1.id

        script_after_second_publish = await async_db.get(Script, script_id)
        assert script_after_second_publish is not None
        assert script_after_second_publish.current_version_id == version_2.id

        count_stmt = (
            select(func.count())
            .select_from(ScriptVersion)
            .where(ScriptVersion.script_id == script_id)
        )
        count_result = await async_db.execute(count_stmt)
        assert count_result.scalar() == 2, "Expected exactly two ScriptVersion rows after two publishes"

        refetched_version_1 = await async_db.get(ScriptVersion, version_1.id)
        refetched_version_2 = await async_db.get(ScriptVersion, version_2.id)
        assert refetched_version_1 is not None
        assert refetched_version_2 is not None
        assert refetched_version_1.version_number == 1
        assert refetched_version_2.version_number == 2
        assert refetched_version_1.content != refetched_version_2.content
        assert refetched_version_1.content["opening_response"] == CONTRACT_V1["opening_response"]
        assert refetched_version_2.content["opening_response"] == CONTRACT_V2["opening_response"]

        # 5. unpublish -> status="unpublished", current_version_id retained
        # (still version 2's id), and get_active_published_version returns
        # None (not consumable).
        unpublished_script = await unpublish(async_db, script_id=script_id, admin_id=admin.id)
        assert unpublished_script.status == ScriptStatus.UNPUBLISHED.value == "unpublished"
        assert unpublished_script.current_version_id == version_2.id

        refetched_script_after_unpublish = await async_db.get(Script, script_id)
        assert refetched_script_after_unpublish is not None
        assert refetched_script_after_unpublish.status == ScriptStatus.UNPUBLISHED.value
        assert refetched_script_after_unpublish.current_version_id == version_2.id

        active_version = await get_active_published_version(async_db, scenario.id)
        assert active_version is None, (
            "Expected an unpublished script to not be consumable via "
            "get_active_published_version"
        )

        # 6. delete_script -> is_deleted=True, get_script returns None, and
        # both ScriptVersion rows (1 and 2) still exist untouched in the DB.
        await delete_script(async_db, script_id=script_id, admin_id=admin.id)

        deleted_script_direct = await async_db.get(Script, script_id)
        assert deleted_script_direct is not None
        assert deleted_script_direct.is_deleted is True

        assert await get_script(async_db, script_id) is None, (
            "Expected get_script to return None for a soft-deleted script"
        )

        final_count_result = await async_db.execute(count_stmt)
        assert final_count_result.scalar() == 2, (
            "Expected both ScriptVersion rows to remain untouched after "
            "soft-deleting the script"
        )
        final_version_1 = await async_db.get(ScriptVersion, version_1.id)
        final_version_2 = await async_db.get(ScriptVersion, version_2.id)
        assert final_version_1 is not None and final_version_1.version_number == 1
        assert final_version_2 is not None and final_version_2.version_number == 2
        assert final_version_1.content["opening_response"] == CONTRACT_V1["opening_response"]
        assert final_version_2.content["opening_response"] == CONTRACT_V2["opening_response"]
