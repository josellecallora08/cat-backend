"""Property-based tests for the Script_Registry lifecycle service.

Feature: ai-debtor-script-contract

This module is shared/appended to across several sub-tasks (7.2, 7.4, 7.6,
7.8), each contributing one test class for one of the correctness
properties defined in `design.md`:

    - Property 11 (this task, 7.2): New scripts always start as
      Draft_Script
    - Property 13 (task 7.4): Editing a Published_Script never alters
      existing Script_Versions
    - Property 12 (task 7.6): Publish succeeds if and only if the draft
      is valid, with no partial side effects
    - Property 14 (task 7.8): Unpublishing preserves versions and blocks
      new consumption

The async in-memory SQLite fixture (`async_db`) and `User`/`Scenario` FK
prerequisite-row helpers below follow the conventions established in
`test_session_service.py`/`test_scenario_repository.py`. The Hypothesis
strategies for generating varied-but-structurally-valid Script_Contract
dicts are adapted from `test_property_script_contract_structure.py`'s/
`test_property_script_limits.py`'s shared `valid_script_contract_dicts`
base, reused here (via `raw_definition_and_format_cases`) so `create_draft`
can be exercised with realistic JSON/YAML content. Later test classes
appended to this file should reuse these strategies/helpers rather than
redefining their own copies.
"""

import json
import uuid

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Scenario, Script, ScriptVersion, User
from app.models.script import ScriptStatus
from app.models.user import UserRole
from app.schemas.script import ScriptContract
from app.services.script_registry import create_draft, publish, unpublish, update_draft
from app.services.script_validator import ScriptValidationError, parse_script_definition

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


# --- Shared strategies (adapted from test_property_script_contract_structure.py /
# test_property_script_limits.py) ---

# Free-text strategy: printable letters/numbers plus a few safe punctuation
# characters, with leading/trailing whitespace stripped so the value is
# guaranteed non-empty and well-formed after stripping. ASCII-restricted
# (rather than full Unicode "L"/"N" categories) to match the alphabet used
# by this spec's other property tests.
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

# Names/scenario labels: shorter safe text, used for the `name` argument to
# `create_draft` (not itself part of the Script_Contract).
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
    rule (Property 4), which is out of scope for this module's lifecycle
    properties.
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


# The ten required top-level Script_Contract fields (Requirement 1.1),
# reused from test_property_script_contract_structure.py's
# TOP_LEVEL_REQUIRED_FIELDS to build a structurally invalid draft for
# TestPublishCorrectness's failure case below.
TOP_LEVEL_REQUIRED_FIELDS = [
    "debtor_persona",
    "financial_situation",
    "opening_response",
    "expected_replies",
    "trigger_phrases",
    "emotional_state_rules",
    "payment_conditions",
    "escalation_conditions",
    "prohibited_responses",
    "conversation_goal",
]


@st.composite
def raw_definition_and_format_cases(draw):
    """Generate a (name, raw_definition, format) case for `create_draft`.

    Serializes a varied, structurally valid Script_Contract-shaped dict to
    either JSON or YAML text, so `create_draft`'s parse -> structural
    validation pipeline is exercised with realistic varied content and
    varied declared format, per Property 11's "regardless of the script's
    content" clause.
    """
    contract_dict = draw(valid_script_contract_dicts())
    name = draw(safe_name_text)
    format = draw(st.sampled_from(["json", "yaml"]))

    if format == "json":
        raw_definition = json.dumps(contract_dict)
    else:
        raw_definition = yaml.safe_dump(contract_dict)

    return name, raw_definition, format


# --- Property Tests ---


class TestNewScriptsStartAsDraft:
    """Property 11: New scripts always start as Draft_Script.

    Feature: ai-debtor-script-contract, Property 11: New scripts always
    start as Draft_Script

    For any valid script creation request submitted by an Administrator,
    the resulting Script's status SHALL be "draft" immediately after
    creation, regardless of the script's content.

    **Validates: Requirements 3.6**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(case=raw_definition_and_format_cases())
    async def test_create_draft_always_persists_draft_status(
        self, async_db: AsyncSession, case
    ):
        """For any varied but structurally valid script creation request,
        `create_draft` SHALL persist a `Script` row whose `status` is
        `"draft"` immediately after creation, regardless of the varied
        name/scenario/content submitted."""
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

        assert script.status == ScriptStatus.DRAFT.value == "draft", (
            f"Expected newly created Script to have status "
            f"{ScriptStatus.DRAFT.value!r}, got {script.status!r}"
        )

        # Verify the persisted row (not just the in-memory return value)
        # also reflects the draft status immediately after creation.
        await async_db.refresh(script)
        assert script.status == ScriptStatus.DRAFT.value

        # Cleanup for next generated example.
        await async_db.rollback()


class TestPublishedScriptEditIsolation:
    """Property 13: Editing a Published_Script never alters existing Script_Versions.

    Feature: ai-debtor-script-contract, Property 13: Editing a
    Published_Script never alters existing Script_Versions

    For any Published_Script with one or more existing Script_Versions,
    editing the script (creating/updating its draft revision) SHALL leave
    the content of every previously created Script_Version byte-for-byte
    unchanged.

    **Validates: Requirements 3.9**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        original_content=valid_script_contract_dicts(),
        original_format=st.sampled_from(["json", "yaml"]),
        edit_case=raw_definition_and_format_cases(),
    )
    async def test_update_draft_leaves_published_version_untouched(
        self, async_db: AsyncSession, original_content, original_format, edit_case
    ):
        """For any Published_Script with an existing ScriptVersion, calling
        `update_draft` with new varied content SHALL update the Script's
        draft_content while leaving `current_version_id` and the existing
        ScriptVersion's `content` byte-for-byte/deep-equal unchanged."""
        _new_name, new_raw_definition, new_format = edit_case

        admin = _make_admin_user()
        publisher = _make_admin_user(name="Test Publisher")
        scenario = _make_scenario()
        async_db.add_all([admin, publisher, scenario])
        await async_db.flush()

        script = Script(
            scenario_id=scenario.id,
            name="Published Script Under Test",
            status=ScriptStatus.PUBLISHED.value,
            format=original_format,
            draft_content=original_content,
            created_by=admin.id,
        )
        async_db.add(script)
        await async_db.flush()

        version = ScriptVersion(
            script_id=script.id,
            version_number=1,
            content=original_content,
            published_by=publisher.id,
        )
        async_db.add(version)
        await async_db.flush()

        script.current_version_id = version.id
        await async_db.commit()
        await async_db.refresh(script)
        await async_db.refresh(version)

        version_id = version.id
        # Record the ScriptVersion's content before the edit, deep-copied
        # via JSON round-trip so later mutation of `script.draft_content`
        # in-place cannot accidentally alias/affect this recorded snapshot.
        recorded_version_content = json.loads(json.dumps(version.content))

        updated_script = await update_draft(
            async_db,
            script_id=script.id,
            raw_definition=new_raw_definition,
            format=new_format,
        )

        expected_new_content = parse_script_definition(new_raw_definition, new_format)
        assert updated_script.draft_content == expected_new_content, (
            "Expected update_draft to overwrite draft_content with the new "
            "edited content"
        )

        assert updated_script.current_version_id == version_id, (
            "Expected editing a Published_Script to leave current_version_id "
            "unchanged"
        )

        # Re-fetch the ScriptVersion row directly to confirm it was not
        # touched by the edit (not just relying on the in-memory object).
        refetched_version = await async_db.get(ScriptVersion, version_id)
        assert refetched_version is not None
        assert refetched_version.content == recorded_version_content, (
            "Expected editing a Published_Script to leave every existing "
            "Script_Version's content byte-for-byte unchanged, but it was "
            f"altered: {refetched_version.content!r} != "
            f"{recorded_version_content!r}"
        )
        assert refetched_version.version_number == 1
        assert refetched_version.script_id == script.id

        # Cleanup for next generated example.
        await async_db.rollback()


class TestPublishCorrectness:
    """Property 12: Publish succeeds if and only if the draft is valid,
    with no partial side effects.

    Feature: ai-debtor-script-contract, Property 12: Publish succeeds if
    and only if the draft is valid, with no partial side effects

    For any Draft_Script, publishing it SHALL succeed (creating exactly
    one new Script_Version whose content matches the draft, and setting
    the Script's status to "published" and current_version_id to the new
    version) if and only if the draft satisfies the Script_Contract and
    all configured limits; if it does not satisfy them, publishing SHALL
    fail, return the validation errors, and SHALL NOT create a
    Script_Version or change the Script's status.

    **Validates: Requirements 3.7, 3.8**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        contract_dict=valid_script_contract_dicts(),
        format=st.sampled_from(["json", "yaml"]),
    )
    async def test_publish_valid_draft_succeeds_with_matching_version(
        self, async_db: AsyncSession, contract_dict, format
    ):
        """For any valid Draft_Script (well within default configured
        limits), `publish` SHALL create exactly one new ScriptVersion with
        `version_number=1` whose content matches the draft, and SHALL set
        the Script's `status` to `"published"` and `current_version_id` to
        that new version's id."""
        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        if format == "json":
            raw_definition = json.dumps(contract_dict)
        else:
            raw_definition = yaml.safe_dump(contract_dict)

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name="Publish Correctness Draft",
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )
        script_id = script.id

        # Expected persisted version content: the same normalization
        # `publish` itself applies (parse -> ScriptContract -> model_dump
        # in JSON mode), so e.g. Decimal `outstanding_balance` values
        # compare correctly regardless of str/float representation.
        expected_content = ScriptContract(**contract_dict).model_dump(mode="json")

        new_version = await publish(async_db, script_id=script_id, admin_id=admin.id)

        assert new_version.version_number == 1, (
            f"Expected the first published version to be version_number=1, "
            f"got {new_version.version_number!r}"
        )
        assert new_version.script_id == script_id
        assert new_version.content == expected_content, (
            "Expected the new ScriptVersion's content to deep-equal the "
            f"draft's validated/normalized content, got: "
            f"{new_version.content!r} != {expected_content!r}"
        )

        # Re-fetch from DB (not just the in-memory return value) to
        # confirm the ScriptVersion was actually persisted.
        refetched_version = await async_db.get(ScriptVersion, new_version.id)
        assert refetched_version is not None
        assert refetched_version.content == expected_content
        assert refetched_version.version_number == 1

        # Exactly one ScriptVersion row exists for this script.
        count_stmt = select(func.count()).select_from(ScriptVersion).where(
            ScriptVersion.script_id == script_id
        )
        count_result = await async_db.execute(count_stmt)
        assert count_result.scalar() == 1, (
            "Expected exactly one ScriptVersion to exist for this script "
            "after a single successful publish"
        )

        # Re-fetch the Script from DB to confirm status/current_version_id
        # were actually persisted, not just set in-memory.
        refetched_script = await async_db.get(Script, script_id)
        assert refetched_script is not None
        assert refetched_script.status == ScriptStatus.PUBLISHED.value == "published", (
            f"Expected Script.status to be 'published' after a successful "
            f"publish, got {refetched_script.status!r}"
        )
        assert refetched_script.current_version_id == new_version.id, (
            "Expected Script.current_version_id to point at the newly "
            f"created ScriptVersion, got {refetched_script.current_version_id!r} "
            f"!= {new_version.id!r}"
        )

        # Cleanup for next generated example.
        await async_db.rollback()

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        contract_dict=valid_script_contract_dicts(),
        format=st.sampled_from(["json", "yaml"]),
        field_to_remove=st.sampled_from(TOP_LEVEL_REQUIRED_FIELDS),
    )
    async def test_publish_invalid_draft_fails_with_no_side_effects(
        self, async_db: AsyncSession, contract_dict, format, field_to_remove
    ):
        """For any Draft_Script whose draft content is missing a required
        top-level Script_Contract field, `publish` SHALL raise
        `ScriptValidationError`, and SHALL NOT create a ScriptVersion or
        change the Script's `status`/`current_version_id` from their
        pre-publish draft state."""
        invalid_contract_dict = dict(contract_dict)
        del invalid_contract_dict[field_to_remove]

        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        # Persist the invalid draft directly via the DB session, bypassing
        # `create_draft`'s own structural validation (which would reject
        # this content outright before it could ever reach `publish`).
        script = Script(
            scenario_id=scenario.id,
            name="Publish Correctness Invalid Draft",
            status=ScriptStatus.DRAFT.value,
            format=format,
            draft_content=invalid_contract_dict,
            created_by=admin.id,
            current_version_id=None,
        )
        async_db.add(script)
        await async_db.commit()
        await async_db.refresh(script)
        script_id = script.id

        with pytest.raises(ScriptValidationError):
            await publish(async_db, script_id=script_id, admin_id=admin.id)

        # Re-fetch the Script from DB to confirm no partial side effects
        # were persisted: status/current_version_id must remain exactly
        # as they were pre-publish.
        refetched_script = await async_db.get(Script, script_id)
        assert refetched_script is not None
        assert refetched_script.status == ScriptStatus.DRAFT.value == "draft", (
            "Expected a failed publish to leave Script.status unchanged as "
            f"'draft', got {refetched_script.status!r}"
        )
        assert refetched_script.current_version_id is None, (
            "Expected a failed publish to leave Script.current_version_id "
            f"unchanged as None, got {refetched_script.current_version_id!r}"
        )

        # No ScriptVersion row must have been created for this script.
        count_stmt = select(func.count()).select_from(ScriptVersion).where(
            ScriptVersion.script_id == script_id
        )
        count_result = await async_db.execute(count_stmt)
        assert count_result.scalar() == 0, (
            "Expected a failed publish to create zero ScriptVersion rows, "
            f"got {count_result.scalar()}"
        )

        # Cleanup for next generated example.
        await async_db.rollback()


class TestUnpublishPreservesVersions:
    """Property 14: Unpublishing preserves versions and blocks new consumption.

    Feature: ai-debtor-script-contract, Property 14: Unpublishing
    preserves versions and blocks new consumption

    For any Published_Script, unpublishing it SHALL leave all of its
    existing Script_Versions unchanged in the database, and SHALL cause
    any subsequent attempt to start a Training_Call against its
    associated scenario to fail as if no Published_Script existed.

    **Validates: Requirements 3.10**

    Scope note: `get_active_published_version` (the function
    Training_Call/session creation will use to check for a consumable
    Published_Script) is task 7.9, not yet implemented, and
    `create_session`'s wiring to it is task 11.1, also not yet
    implemented. So this test class exercises only what IS implementable
    now with `unpublish` (task 7.7):

      1. The "versions preserved" half, fully: every existing
         ScriptVersion row's fields (content, version_number,
         published_by, published_at) and Script.current_version_id are
         asserted byte-for-byte/deep-equal unchanged after unpublishing.
      2. A proxy for the "blocks new consumption" half: asserting
         Script.status == "unpublished" (!= "published"), since task
         7.9's `get_active_published_version` will be defined to only
         return a version when `status == "published"` — this status
         flip is the mechanism that will cause future consumption to
         fail. The full end-to-end "Training_Call fails to start"
         behavior is exercised later by task 7.10 (Property 15, in
         `test_property_script_consumption.py`) once
         `get_active_published_version` exists.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        contract_dict=valid_script_contract_dicts(),
        format=st.sampled_from(["json", "yaml"]),
    )
    async def test_unpublish_preserves_versions_and_flips_status(
        self, async_db: AsyncSession, contract_dict, format
    ):
        """For any Published_Script with an existing ScriptVersion,
        calling `unpublish` SHALL set `Script.status` to `"unpublished"`
        while leaving `Script.current_version_id` and every field of the
        existing ScriptVersion row (content, version_number,
        published_by, published_at) completely unchanged."""
        admin = _make_admin_user()
        scenario = _make_scenario()
        async_db.add_all([admin, scenario])
        await async_db.flush()

        if format == "json":
            raw_definition = json.dumps(contract_dict)
        else:
            raw_definition = yaml.safe_dump(contract_dict)

        script = await create_draft(
            async_db,
            admin_id=admin.id,
            name="Unpublish Property Draft",
            scenario_id=scenario.id,
            format=format,
            raw_definition=raw_definition,
        )
        script_id = script.id

        new_version = await publish(async_db, script_id=script_id, admin_id=admin.id)
        version_id = new_version.id

        # Confirm pre-unpublish state: published, with the version wired
        # up as current_version_id.
        pre_script = await async_db.get(Script, script_id)
        assert pre_script is not None
        assert pre_script.status == ScriptStatus.PUBLISHED.value == "published"
        assert pre_script.current_version_id == version_id

        # Record the ScriptVersion's full field state before unpublishing,
        # deep-copying `content` via a JSON round-trip so later mutation
        # cannot accidentally alias/affect this recorded snapshot.
        pre_version = await async_db.get(ScriptVersion, version_id)
        assert pre_version is not None
        recorded_content = json.loads(json.dumps(pre_version.content))
        recorded_version_number = pre_version.version_number
        recorded_published_by = pre_version.published_by
        recorded_published_at = pre_version.published_at

        updated_script = await unpublish(async_db, script_id=script_id, admin_id=admin.id)

        assert updated_script.status == ScriptStatus.UNPUBLISHED.value == "unpublished", (
            f"Expected Script.status to be 'unpublished' after unpublish, "
            f"got {updated_script.status!r}"
        )
        assert updated_script.current_version_id == version_id, (
            "Expected unpublish to leave Script.current_version_id "
            f"unchanged, got {updated_script.current_version_id!r} != "
            f"{version_id!r}"
        )

        # Re-fetch the Script from DB (not just the in-memory return
        # value) to confirm the status flip and unchanged
        # current_version_id were actually persisted.
        refetched_script = await async_db.get(Script, script_id)
        assert refetched_script is not None
        assert refetched_script.status == ScriptStatus.UNPUBLISHED.value == "unpublished", (
            "Expected the persisted Script.status to be 'unpublished' "
            f"(!= 'published', the value get_active_published_version will "
            f"require), got {refetched_script.status!r}"
        )
        assert refetched_script.current_version_id == version_id

        # Re-fetch the ScriptVersion row directly to confirm every field
        # is byte-for-byte/deep-equal unchanged from before unpublishing.
        refetched_version = await async_db.get(ScriptVersion, version_id)
        assert refetched_version is not None
        assert refetched_version.content == recorded_content, (
            "Expected unpublish to leave the existing Script_Version's "
            f"content unchanged, got {refetched_version.content!r} != "
            f"{recorded_content!r}"
        )
        assert refetched_version.version_number == recorded_version_number
        assert refetched_version.published_by == recorded_published_by
        assert refetched_version.published_at == recorded_published_at
        assert refetched_version.script_id == script_id

        # Cleanup for next generated example.
        await async_db.rollback()
