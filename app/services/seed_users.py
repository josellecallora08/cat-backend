"""Default user seeder for the Collection Agent Trainer.

Seeds the database with default admin and agent accounts on startup.
Only inserts users that don't already exist (matched by email).
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, UserRole
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
        "role": UserRole.AGENT.value,
    },
]


async def seed_default_users(db: AsyncSession) -> None:
    """Seed the database with default user accounts.

    Only inserts users whose emails don't already exist in the database,
    making this safe to call on every startup.

    Default accounts:
      - admin@cat.ph / admin123 (Administrator)
      - agent@cat.ph / agent123 (Collection Agent)

    Args:
        db: An async database session.
    """
    # Get existing user emails
    stmt = select(User.email)
    result = await db.execute(stmt)
    existing_emails = set(result.scalars().all())

    inserted = 0
    for user_data in DEFAULT_USERS:
        if user_data["email"] in existing_emails:
            continue

        user = User(
            id=uuid.uuid4(),
            email=user_data["email"],
            hashed_password=hash_password(user_data["password"]),
            full_name=user_data["full_name"],
            role=user_data["role"],
        )
        db.add(user)
        inserted += 1

    if inserted > 0:
        await db.commit()
        logger.info("Seeded %d default users", inserted)
    else:
        logger.info("All default users already exist, skipping seed")
