"""Property-based tests for Published_Script consumption by Training_Call.

Feature: ai-debtor-script-contract

This module is shared/appended to across sub-tasks (7.10, 11.2), each
contributing one test class for one of the correctness properties defined
in `design.md`:

    - Property 15 (7.10): Training_Call loading is restricted to the
      Published_Script's current version
    - Property 16 (this task, 11.2): Training_Call start fails
      descriptively without a Published_Script

The async in-memory SQLite fixture (`async_db`) and `User`/`Scenario` FK
prerequisite-row helpers below follow the conventions established in
`test_property_script_lifecycle.py`. Later test classes appended to this
file should reuse these fixtures/helpers rather than redefining their own
copies.

Scope note for this task (7.10): `create_session`/Training_Call creation
itself is not yet wired to `get_active_published_version` (that's task
11.1, not yet implemented). So this test exercises
`get_active_published_version` directly, as the mechanism Training_Call
creation will depend on to enforce this property.
"""

import json
import uuid
from unittest.mock import AsyncMock

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import event, select, text

from app.database import Base
from app.models import Scenario, Script, ScriptVersion, Session, User
from app.models.script import ScriptStatus
from app.models.user import UserRole
from app.services.debtor_simulator import DebtorSimulatorService, EmotionalState, PersonaContext
from app.services.script_registry import create_draft, get_active_published_version, publish, unpublish
from app.services.session_service import create_session

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
    # rows populate both sides (e.g. simulating a Published_Script), SQLite's
    # per-statement FK checking can otherwise reject DROP TABLE regardless of
    # drop order.
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys=OFF"))
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# --- FK prerequisite-row helpers (shared by later sub-tasks appended here) ---


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
        description="A test scenario for script consumption tests",
        debtor_profile={
            "name": "Maria Santos",
            "outstanding_balance": "5000.00",
            "days_past_due": 45,
            "personality_profile": "anxious",
            "conversation_goal": "negotiate payment plan",
        },
        is_active=True,
    )


# --- Shared strategies (adapted from test_property_script_lifecycle.py) ---

safe_text = (
    st.text(
        alphabet=st.characters(
            whitelist_categories=(),
            whitelist_characters=(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"
            ),
        ),
        min_size=1,
        max_size=50,
    )
    .map(lambda s: s.strip())
    .filter(lambda s: s != "")
)

safe_name_text = (
    st.text(
        alphabet=st.characters(
            whitelist_categories=(),
            whitelist_characters=(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_"
            ),
        ),
        min_size=1,
        max_size=40,
    )
    .map(lambda s: s.strip())
    .filter(lambda s: s != "")
)

trigger_phrase_entries = st.fixed_dictionaries({"phrase": safe_text, "behavior": safe_text})

expected_reply_entries = st.fixed_dictionaries(
    {"agent_statement": safe_text, "debtor_reply": safe_text}
)

emotional_state_rule_entries = st.fixed_dictionaries({"trigger": safe_text, "state_change": safe_text})

escalation_condition_entries = st.fixed_dictionaries(
    {"condition": safe_text, "behavior": safe_text, "ends_call": st.booleans()}
)

payment_condition_entries = st.fixed_dictionaries(
    {"condition": safe_text, "term": safe_text, "accepted": st.booleans()}
)

debtor_persona_dicts = st.fixed_dictionaries(
    {"name": safe_text, "communication_style": safe_text, "background": safe_text}
)

financial_situation_dicts = st.fixed_dictionaries(
    {
        "outstanding_balance": st.floats(
            min_value=0.01,
            max_value=999_999.0,
            allow_nan=False,
            allow_infinity=False,
        ).map(lambda x: round(x, 2)),
        "days_past_due": st.integers(min_value=0, max_value=10_000),
        "reason_for_delinquency": safe_text,
    }
)

conversation_goal_dicts = st.fixed_dictionaries(
    {"target_outcome": safe_text, "completion_condition": safe_text}
)


@st.composite
def valid_script_contract_dicts(draw):
    """Generate a structurally valid, varied Script_Contract-shaped dict.

    `prohibited_responses` is kept empty so no generated example
    accidentally triggers the (unrelated) Prohibited/Expected conflict
    rule (Property 4), which is out of scope for this module's
    consumption properties.
    """
    return {
        "debtor_persona": draw(debtor_persona_dicts),
        "financial_situation": draw(financial_situation_dicts),
        "opening_response": draw(safe_text),
        "expected_replies": draw(st.lists(expected_reply_entries, min_size=0, max_size=5)),
        "trigger_phrases": draw(st.lists(trigger_phrase_entries, min_size=0, max_size=5)),
        "emotional_state_rules": draw(
            st.lists(emotional_state_rule_entries, min_size=0, max_size=5)
        ),
        "payment_conditions": draw(st.lists(payment_condition_entries, min_size=0, max_size=5)),
        "escalation_conditions": draw(
            st.lists(escalation_condition_entries, min_size=0, max_size=5)
        ),
        "prohibited_responses": [],
        "conversation_goal": draw(conversation_goal_dicts),
    }


@st.composite
def raw_definition_and_format_cases(draw):
    """Generate a (name, raw_definition, format) case for `create_draft`."""
    contract_dict = draw(valid_script_contract_dicts())
    name = draw(safe_name_text)
    format = draw(st.sampled_from(["json", "yaml"]))

    if format == "json":
        raw_definition = json.dumps(contract_dict)
    else:
        raw_definition = yaml.safe_dump(contract_dict)

    return name, raw_definition, format


# --- Property Tests ---


class TestRestrictedVersionLoading:
    """Property 15: Training_Call loading is restricted to the
    Published_Script's current version.

    Feature: ai-debtor-script-contract, Property 15: Training_Call
    loading is restricted to the Published_Script's current version

    For any scenario, starting a Training_Call SHALL load the current
    Script_Version of that scenario's Published_Script when one exists
    (and only when one exists -- a scenario whose associated script is
    only ever a Draft_Script, or has been unpublished, or does not
    exist, behaves identically to having no Published_Script).

    **Validates: Requirements 4.1, 4.3**

    Scope note: `create_session`/Training_Call creation itself is not
    yet wired to `get_active_published_version` (task 11.1, not yet
    implemented), so this test exercises `get_active_published_version`
    directly -- the mechanism this property depends on.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=st.just(None))
    async def test_no_script_returns_none(self, async_db: AsyncSession, case):
        """For any scenario with no associated `Script` row at all,
        `get_active_published_version` SHALL return `None`."""
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        result = await get_active_published_version(async_db, scenario.id)

        assert result is None, (
            "Expected get_active_published_version to return None when no "
            f"Script exists for the scenario, got {result!r}"
        )

        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=raw_definition_and_format_cases())
    async def test_draft_only_script_returns_none(self, async_db: AsyncSession, case):
        """For any scenario whose associated `Script` exists only as a
        Draft_Script (never published), `get_active_published_version`
        SHALL return `None`, regardless of the draft's varied content."""
        name, raw_definition, format = case

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        await create_draft(
            async_db,
            admin_id=admin.id,
            name=name,
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )

        result = await get_active_published_version(async_db, scenario.id)

        assert result is None, (
            "Expected get_active_published_version to return None for a "
            f"draft-only Script, got {result!r}"
        )

        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=raw_definition_and_format_cases())
    async def test_published_script_returns_matching_current_version(
        self, async_db: AsyncSession, case
    ):
        """For any scenario whose associated `Script` is published,
        `get_active_published_version` SHALL return the `ScriptVersion`
        whose id and content match the `Script`'s `current_version_id`
        and the just-published content."""
        name, raw_definition, format = case

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name=name,
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )

        published_version = await publish(async_db, script_id=script.id, admin_id=admin.id)

        result = await get_active_published_version(async_db, scenario.id)

        assert result is not None, (
            "Expected get_active_published_version to return a ScriptVersion "
            "for a published Script, got None"
        )
        assert result.id == published_version.id, (
            f"Expected the returned ScriptVersion's id to match the "
            f"published version's id, got {result.id!r} != "
            f"{published_version.id!r}"
        )
        assert result.content == published_version.content, (
            "Expected the returned ScriptVersion's content to match the "
            f"published version's content, got {result.content!r} != "
            f"{published_version.content!r}"
        )

        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=raw_definition_and_format_cases())
    async def test_unpublished_script_returns_none_despite_current_version_id(
        self, async_db: AsyncSession, case
    ):
        """For any scenario whose associated `Script` was published then
        unpublished, `get_active_published_version` SHALL return `None`,
        despite `current_version_id` still being set and the
        `ScriptVersion` row still existing in the database."""
        name, raw_definition, format = case

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name=name,
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )

        published_version = await publish(async_db, script_id=script.id, admin_id=admin.id)
        version_id = published_version.id

        await unpublish(async_db, script_id=script.id, admin_id=admin.id)

        result = await get_active_published_version(async_db, scenario.id)

        assert result is None, (
            "Expected get_active_published_version to return None for an "
            f"unpublished Script (despite current_version_id still being "
            f"set), got {result!r}"
        )

        # Confirm current_version_id is still set and the ScriptVersion
        # row still exists -- the "despite" part of this property.
        refetched_script = await async_db.get(Script, script.id)
        assert refetched_script is not None
        assert refetched_script.current_version_id == version_id, (
            "Expected current_version_id to remain set after unpublishing, "
            f"got {refetched_script.current_version_id!r}"
        )
        refetched_version = await async_db.get(ScriptVersion, version_id)
        assert refetched_version is not None, (
            "Expected the ScriptVersion row to still exist after "
            "unpublishing"
        )

        await async_db.rollback()


# --- Helpers shared by Property 16 tests below ---


def _make_mock_debtor_simulator() -> DebtorSimulatorService:
    """Create a mock DebtorSimulatorService that returns a valid persona.

    Mirrors `test_session_service.py`'s `_make_mock_debtor_simulator`
    helper. `create_session` calls `generate_persona` as part of its
    persona-generation step; for these tests we only care about the
    Published_Script gate raising before any Session row is created, but
    the simulator still needs to behave like a real one in case the gate
    were absent.
    """
    mock_llm = AsyncMock()
    simulator = DebtorSimulatorService(llm_service=mock_llm)

    mock_persona = PersonaContext(
        persona_id=uuid.uuid4(),
        name="Maria Santos",
        communication_style="anxious",
        financial_circumstances={
            "income_level": "low",
            "debt_amount": 5000,
            "reason_for_delinquency": "job loss",
        },
        emotional_state=EmotionalState.DEFENSIVE,
        language="EN",
    )
    simulator.generate_persona = AsyncMock(return_value=mock_persona)
    return simulator


class TestTrainingCallFailsWithoutPublishedScript:
    """Property 16: Training_Call start fails descriptively without a
    Published_Script.

    Feature: ai-debtor-script-contract, Property 16: Training_Call start
    fails descriptively without a Published_Script

    For any scenario with no Published_Script (no script at all,
    draft-only, or unpublished), attempting to start a Training_Call
    SHALL fail with a descriptive error and SHALL NOT create a Session
    with a script_version_id.

    **Validates: Requirements 4.2**
    """

    async def _assert_create_session_fails_without_partial_session(
        self, async_db: AsyncSession, scenario_id: uuid.UUID
    ) -> None:
        """Shared assertion: create_session raises ValueError and no
        Session row exists for scenario_id afterward."""
        simulator = _make_mock_debtor_simulator()
        agent_id = uuid.uuid4()

        with pytest.raises(ValueError) as exc_info:
            await create_session(async_db, scenario_id, agent_id, simulator)

        # Descriptive: the error must say something more than nothing.
        assert str(exc_info.value).strip() != "", (
            "Expected create_session to raise a ValueError with a "
            "descriptive (non-empty) message when no Published_Script "
            "exists"
        )

        result = await async_db.execute(
            select(Session).where(Session.scenario_id == scenario_id)
        )
        sessions = result.scalars().all()
        assert sessions == [], (
            "Expected no Session row to be created for the scenario when "
            f"create_session fails due to a missing Published_Script, "
            f"found {len(sessions)} row(s)"
        )

        await async_db.rollback()

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=st.just(None))
    async def test_no_script_at_all_fails_without_partial_session(
        self, async_db: AsyncSession, case
    ):
        """For any scenario with no associated `Script` row at all,
        `create_session` SHALL raise a descriptive `ValueError` and
        SHALL NOT create a `Session` row for that scenario."""
        scenario = _make_scenario()
        async_db.add(scenario)
        await async_db.commit()

        await self._assert_create_session_fails_without_partial_session(
            async_db, scenario.id
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=raw_definition_and_format_cases())
    async def test_draft_only_script_fails_without_partial_session(
        self, async_db: AsyncSession, case
    ):
        """For any scenario whose associated `Script` exists only as a
        Draft_Script (never published), `create_session` SHALL raise a
        descriptive `ValueError` and SHALL NOT create a `Session` row for
        that scenario, regardless of the draft's varied content."""
        name, raw_definition, format = case

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()
        await async_db.commit()

        await create_draft(
            async_db,
            admin_id=admin.id,
            name=name,
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )

        await self._assert_create_session_fails_without_partial_session(
            async_db, scenario.id
        )

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=raw_definition_and_format_cases())
    async def test_unpublished_script_fails_without_partial_session(
        self, async_db: AsyncSession, case
    ):
        """For any scenario whose associated `Script` was published then
        unpublished, `create_session` SHALL raise a descriptive
        `ValueError` and SHALL NOT create a `Session` row for that
        scenario, despite the `Script`'s `current_version_id` and
        `ScriptVersion` row still existing."""
        name, raw_definition, format = case

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()
        await async_db.commit()

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name=name,
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )
        await publish(async_db, script_id=script.id, admin_id=admin.id)
        await unpublish(async_db, script_id=script.id, admin_id=admin.id)

        await self._assert_create_session_fails_without_partial_session(
            async_db, scenario.id
        )
