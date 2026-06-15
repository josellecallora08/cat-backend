"""Property-based tests for DebtorProfileSchema validation.

Feature: collection-agent-trainer, Property 2: Incomplete profile rejection

**Validates: Requirements 1.6**

Property 2: For any debtor profile where one or more of the required fields
(name, outstanding_balance, days_past_due, personality_profile, conversation_goal)
is missing or empty, the profile validation SHALL reject it and produce an error.
"""

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.schemas import DebtorProfileSchema


# --- Strategies ---

# Valid field strategies for constructing complete profiles
valid_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() != "")

valid_balances = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("999999.99"),
    allow_nan=False,
    allow_infinity=False,
    places=2,
)

valid_days_past_due = st.integers(min_value=1, max_value=10000)

valid_personality_profiles = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

valid_conversation_goals = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

# Invalid string strategies: empty or whitespace-only
empty_or_whitespace_strings = st.one_of(
    st.just(""),
    st.text(alphabet=st.just(" "), min_size=1, max_size=20),
    st.text(alphabet=st.sampled_from([" ", "\t", "\n", "\r"]), min_size=1, max_size=10),
)

# Invalid balance strategies: zero or negative
invalid_balances = st.one_of(
    st.just(Decimal("0")),
    st.decimals(
        min_value=Decimal("-99999.99"),
        max_value=Decimal("0"),
        allow_nan=False,
        allow_infinity=False,
        places=2,
    ),
)

# Invalid days_past_due: zero or negative
invalid_days = st.integers(min_value=-1000, max_value=0)


class TestIncompleteProfileRejection:
    """Property 2: Incomplete profile rejection.

    Feature: collection-agent-trainer, Property 2: Incomplete profile rejection
    """

    @given(invalid_name=empty_or_whitespace_strings)
    @settings(max_examples=100)
    def test_rejects_empty_or_whitespace_name(self, invalid_name: str):
        """Profile with empty/whitespace name is always rejected."""
        with pytest.raises(ValidationError):
            DebtorProfileSchema(
                name=invalid_name,
                outstanding_balance=Decimal("1000.00"),
                days_past_due=30,
                personality_profile="cooperative",
                conversation_goal="negotiate payment",
            )

    @given(invalid_balance=invalid_balances)
    @settings(max_examples=100)
    def test_rejects_zero_or_negative_balance(self, invalid_balance: Decimal):
        """Profile with zero or negative outstanding_balance is always rejected."""
        with pytest.raises(ValidationError):
            DebtorProfileSchema(
                name="John Doe",
                outstanding_balance=invalid_balance,
                days_past_due=30,
                personality_profile="cooperative",
                conversation_goal="negotiate payment",
            )

    @given(invalid_day=invalid_days)
    @settings(max_examples=100)
    def test_rejects_zero_or_negative_days_past_due(self, invalid_day: int):
        """Profile with zero or negative days_past_due is always rejected."""
        with pytest.raises(ValidationError):
            DebtorProfileSchema(
                name="John Doe",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=invalid_day,
                personality_profile="cooperative",
                conversation_goal="negotiate payment",
            )

    @given(invalid_personality=empty_or_whitespace_strings)
    @settings(max_examples=100)
    def test_rejects_empty_or_whitespace_personality(self, invalid_personality: str):
        """Profile with empty/whitespace personality_profile is always rejected."""
        with pytest.raises(ValidationError):
            DebtorProfileSchema(
                name="John Doe",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=30,
                personality_profile=invalid_personality,
                conversation_goal="negotiate payment",
            )

    @given(invalid_goal=empty_or_whitespace_strings)
    @settings(max_examples=100)
    def test_rejects_empty_or_whitespace_conversation_goal(self, invalid_goal: str):
        """Profile with empty/whitespace conversation_goal is always rejected."""
        with pytest.raises(ValidationError):
            DebtorProfileSchema(
                name="John Doe",
                outstanding_balance=Decimal("1000.00"),
                days_past_due=30,
                personality_profile="cooperative",
                conversation_goal=invalid_goal,
            )

    @given(
        field_to_invalidate=st.integers(min_value=0, max_value=4),
        invalid_str=empty_or_whitespace_strings,
        invalid_balance=invalid_balances,
        invalid_day=invalid_days,
    )
    @settings(max_examples=100)
    def test_rejects_any_single_invalid_field(
        self,
        field_to_invalidate: int,
        invalid_str: str,
        invalid_balance: Decimal,
        invalid_day: int,
    ):
        """Profile with any single field invalid is always rejected.

        Randomly selects which field to make invalid and verifies rejection.
        """
        kwargs = {
            "name": "John Doe",
            "outstanding_balance": Decimal("1000.00"),
            "days_past_due": 30,
            "personality_profile": "cooperative",
            "conversation_goal": "negotiate payment",
        }

        if field_to_invalidate == 0:
            kwargs["name"] = invalid_str
        elif field_to_invalidate == 1:
            kwargs["outstanding_balance"] = invalid_balance
        elif field_to_invalidate == 2:
            kwargs["days_past_due"] = invalid_day
        elif field_to_invalidate == 3:
            kwargs["personality_profile"] = invalid_str
        else:
            kwargs["conversation_goal"] = invalid_str

        with pytest.raises(ValidationError):
            DebtorProfileSchema(**kwargs)


class TestValidProfileAcceptance:
    """Complementary property: valid profiles are always accepted."""

    @given(
        name=valid_names,
        balance=valid_balances,
        days=valid_days_past_due,
        personality=valid_personality_profiles,
        goal=valid_conversation_goals,
    )
    @settings(max_examples=100)
    def test_valid_profiles_always_accepted(
        self,
        name: str,
        balance: Decimal,
        days: int,
        personality: str,
        goal: str,
    ):
        """A profile with all valid fields always passes validation."""
        profile = DebtorProfileSchema(
            name=name,
            outstanding_balance=balance,
            days_past_due=days,
            personality_profile=personality,
            conversation_goal=goal,
        )
        assert profile.name == name
        assert profile.outstanding_balance == balance
        assert profile.days_past_due == days
        assert profile.personality_profile == personality
        assert profile.conversation_goal == goal
