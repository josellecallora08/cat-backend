"""Default user seeder for the Collection Agent Trainer.

Seeds the database with default admin and agent accounts on startup.
Only inserts users that don't already exist (matched by email).
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, UserRole, UserType
from app.services.auth import hash_password

logger = logging.getLogger(__name__)

DEFAULT_USERS = [
    {
        "email": "admin@cat.ph",
        "password": "admin123",
        "full_name": "Admin User",
        "role": UserRole.ADMIN.value,
    },
    {
        "email": "agent@cat.ph",
        "password": "agent123",
        "full_name": "Agent User",
        "role": UserRole.USER.value,
        "user_type": UserType.AGENT.value,
    },
]


async def seed_default_users(db: AsyncSession) -> None:
    """Seed the database with default user accounts.

    Inserts users whose emails don't already exist. For existing users,
    updates their password hash to ensure compatibility.

    Default accounts:
      - admin@cat.ph / admin123 (Administrator)
      - agent@cat.ph / agent123 (Collection Agent)

    Args:
        db: An async database session.
    """
    # Get existing user emails
    stmt = select(User)
    result = await db.execute(stmt)
    existing_users = {u.email: u for u in result.scalars().all()}

    changed = False
    for user_data in DEFAULT_USERS:
        if user_data["email"] in existing_users:
            # Update password hash to ensure it works with current bcrypt
            existing = existing_users[user_data["email"]]
            new_hash = hash_password(user_data["password"])
            if existing.hashed_password != new_hash:
                existing.hashed_password = new_hash
                changed = True
            continue

        user = User(
            id=uuid.uuid4(),
            email=user_data["email"],
            hashed_password=hash_password(user_data["password"]),
            full_name=user_data["full_name"],
            role=user_data["role"],
            user_type=user_data.get("user_type"),
        )
        db.add(user)
        changed = True

    if changed:
        await db.commit()
        logger.info("Seeded/updated default users")
    else:
        logger.info("All default users already exist, skipping seed")
