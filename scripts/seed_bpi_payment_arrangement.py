"""Seed the temporary financial hardship scenario into the BPI campaign.

Usage:
    python -m scripts.seed_bpi_payment_arrangement
"""

import asyncio
import logging

from app.database import async_session_factory
from app.services.seed_scenarios import seed_bpi_payment_arrangement_scenario


async def main() -> None:
    """Run the additive, idempotent BPI scenario seed."""
    logging.basicConfig(level=logging.INFO)
    async with async_session_factory() as db:
        result = await seed_bpi_payment_arrangement_scenario(db)
    if result is None:
        logging.getLogger(__name__).error("BPI campaign was not found; seed was not applied.")
        return
    campaign_id, scenario_id = result
    logging.getLogger(__name__).info(
        "Verified seed identifiers: campaign_id=%s scenario_id=%s",
        campaign_id,
        scenario_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
