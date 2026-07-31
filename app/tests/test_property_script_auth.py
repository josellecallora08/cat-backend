"""Property-based test for admin-only enforcement on the Script_Registry API.

Feature: ai-debtor-script-contract

    - Property 10 (task 9.4): Admin-only mutation enforcement

Authorization for every mutating `app/api/scripts.py` endpoint is enforced
entirely at the API layer via `Depends(require_admin)` (see design.md); the
service layer (`script_registry.py`) does not re-check roles itself. This
test therefore exercises the FastAPI app (via `httpx.AsyncClient`, mirroring
`test_campaigns.py`'s conventions) rather than `script_registry.py` directly.

`require_admin` (the dependency under test) is never overridden here, since
overriding it would bypass the very check being validated. Instead, the
*upstream* dependency `get_current_user` is overridden to inject a mock
authenticated user with a controlled `role`, and `require_auth`/
`require_admin` are left to run for real on top of it -- exactly mirroring
how the real JWT-based auth flow reaches `require_admin` in production.
"""

import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.auth import get_current_user

# --- Fixtures ---


@pytest.fixture
async def unauth_client():
    """Async test client with no auth overrides pre-applied.

    Each test installs its own `get_current_user` override (never
    `require_admin` itself -- see module docstring) before issuing a
    request through this client.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Ensure dependency overrides never leak across examples/tests."""
    yield
    app.dependency_overrides.clear()


# --- Helpers ---


def _make_mock_user(role: str) -> MagicMock:
    """Return a mock authenticated `User` with the given role."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "test-user@test.com"
    user.full_name = "Test User"
    user.role = role
    user.is_active = True
    return user


def _make_mock_script(script_id: uuid.UUID) -> MagicMock:
    """Return a mock `Script` ORM object sufficient for response serialization."""
    script = MagicMock()
    script.id = script_id
    script.name = "Test Script"
    script.scenario_id = uuid.uuid4()
    script.status = "draft"
    script.format = "json"
    script.draft_content = {"opening_response": "hi"}
    script.current_version_id = None
    script.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    script.updated_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return script


def _operation_configs(script_id: uuid.UUID) -> dict:
    """Build the per-operation request/patch configuration for all 5
    mutating Script_Registry operations (create, edit, publish, unpublish,
    delete), keyed by operation name.

    Each config's `primary_target` is the service-layer function that
    would perform the actual state change; asserting it was never called
    is how "no Script or Script_Version state changed" is verified for a
    denied request.
    """
    script = _make_mock_script(script_id)
    return {
        "create": {
            "method": "post",
            "url": "/api/scripts",
            "json": {
                "name": "Test Script",
                "scenario_id": str(uuid.uuid4()),
                "format": "json",
                "raw_definition": '{"opening_response": "hi"}',
            },
            "primary_target": "app.api.scripts.create_draft",
            "primary_return": script,
            "extra_targets": [],
        },
        "edit": {
            "method": "put",
            "url": f"/api/scripts/{script_id}",
            "json": {
                "raw_definition": '{"opening_response": "hi"}',
                "format": "json",
            },
            "primary_target": "app.api.scripts.update_draft",
            "primary_return": script,
            "extra_targets": [],
        },
        "publish": {
            "method": "post",
            "url": f"/api/scripts/{script_id}/publish",
            "json": None,
            "primary_target": "app.api.scripts.publish",
            "primary_return": None,
            # `publish_script_endpoint` re-fetches the script via
            # `get_script` to build its response after `publish()`
            # succeeds; patched so a permitted (admin) request doesn't
            # hit a real (nonexistent) database.
            "extra_targets": [("app.api.scripts.get_script", script)],
        },
        "unpublish": {
            "method": "post",
            "url": f"/api/scripts/{script_id}/unpublish",
            "json": None,
            "primary_target": "app.api.scripts.unpublish",
            "primary_return": script,
            "extra_targets": [],
        },
        "delete": {
            "method": "delete",
            "url": f"/api/scripts/{script_id}",
            "json": None,
            "primary_target": "app.api.scripts.delete_script",
            "primary_return": None,
            "extra_targets": [],
        },
    }


OPERATION_KEYS = ["create", "edit", "publish", "unpublish", "delete"]

# Any role string that is not the exact literal "admin" -- `require_admin`
# denies on `user.role != UserRole.ADMIN.value` ("admin"), a plain string
# comparison, so this covers the real "agent" role plus arbitrary garbage
# role strings (including the empty string).
non_admin_roles = st.one_of(
    st.just("agent"),
    st.text(max_size=20).filter(lambda r: r != "admin"),
)


async def _issue_request(client: AsyncClient, config: dict):
    """Issue the HTTP request described by an operation config, patching
    its primary (and any extra) service-layer target(s) for the duration
    of the call. Returns (response, primary_mock)."""
    patches = [
        patch(
            config["primary_target"],
            new_callable=AsyncMock,
            return_value=config["primary_return"],
        )
    ]
    for target, ret in config["extra_targets"]:
        patches.append(patch(target, new_callable=AsyncMock, return_value=ret))

    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        primary_mock = mocks[0]
        request_kwargs = {"json": config["json"]} if config["json"] is not None else {}
        response = await getattr(client, config["method"])(config["url"], **request_kwargs)

    return response, primary_mock


class TestAdminOnlyMutationEnforcement:
    """Property 10: Admin-only mutation enforcement.

    Feature: ai-debtor-script-contract, Property 10: Admin-only mutation
    enforcement

    For any mutating Script_Registry operation (create, edit, publish,
    unpublish, delete) and any User without the Administrator role, the
    operation SHALL be denied with an authorization error and SHALL NOT
    change any Script or Script_Version state; for any User with the
    Administrator role, the operation SHALL be permitted to proceed to
    its business-logic validation.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        operation_key=st.sampled_from(OPERATION_KEYS),
        role=non_admin_roles,
    )
    async def test_non_admin_denied_with_no_state_change(
        self, unauth_client: AsyncClient, operation_key, role
    ):
        """A User whose role is not "admin" is denied with a 401/403 and
        the underlying service-layer mutation function is never invoked
        (so no Script/Script_Version state could have changed)."""
        script_id = uuid.uuid4()
        config = _operation_configs(script_id)[operation_key]

        mock_user = _make_mock_user(role)
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response, primary_mock = await _issue_request(unauth_client, config)

        assert response.status_code in (401, 403), (
            f"Expected operation={operation_key!r} with role={role!r} to be denied "
            f"with 401/403, got {response.status_code}"
        )
        primary_mock.assert_not_called()

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(operation_key=st.sampled_from(OPERATION_KEYS))
    async def test_admin_permitted_to_proceed(
        self, unauth_client: AsyncClient, operation_key
    ):
        """A User whose role is "admin" is not blocked by authorization:
        the request proceeds past `require_admin` into the underlying
        service-layer function (asserted via call count), rather than
        being short-circuited with a 401/403."""
        script_id = uuid.uuid4()
        config = _operation_configs(script_id)[operation_key]

        mock_user = _make_mock_user("admin")
        app.dependency_overrides[get_current_user] = lambda: mock_user

        response, primary_mock = await _issue_request(unauth_client, config)

        assert response.status_code not in (401, 403), (
            f"Expected operation={operation_key!r} with an admin role to proceed past "
            f"authorization, got {response.status_code}"
        )
        primary_mock.assert_called_once()
