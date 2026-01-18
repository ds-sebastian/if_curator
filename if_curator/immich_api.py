"""Immich API client for fetching people and assets."""

import logging
from datetime import datetime, timedelta, timezone

import requests

from .config import Config, get_headers

logger = logging.getLogger(__name__)


def get_people() -> list[dict]:
    """Fetch all people from Immich."""
    try:
        resp = requests.get(
            f"{Config.IMMICH_URL}/api/people",
            headers=get_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("people", [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch people: {e}")
        return []


def fetch_all_assets(person: dict) -> list[dict]:
    """Fetch all assets for a person with pagination."""
    name = person.get("name", "Unknown")
    person_id = person["id"]
    url = f"{Config.IMMICH_URL}/api/search/metadata"
    page_size = 1000

    logger.info(f"Fetching assets for {name}...")

    assets = []
    for page in range(1, 1000):  # Safety limit
        try:
            resp = requests.post(
                url,
                json={"personIds": [person_id], "size": page_size, "page": page},
                headers=get_headers(),
                timeout=30,
            )

            if not resp.ok:
                logger.error(f"Error fetching assets for {name} (page {page}): {resp.status_code}")
                break

            page_assets = resp.json().get("assets", [])
            if isinstance(page_assets, dict):
                page_assets = page_assets.get("items", [])

            if not page_assets:
                break

            assets.extend(page_assets)
            logger.debug(f"Fetched page {page}, total: {len(assets)}")

            if len(page_assets) < page_size:
                break

        except requests.RequestException as e:
            logger.error(f"Exception fetching assets for {name}: {e}")
            break

    return assets


def filter_recent_assets(assets: list[dict], years: int | None = None) -> list[dict]:
    """Filter assets to keep only those from the last N years."""
    years = years or Config.YEARS_FILTER
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * years)

    logger.debug(f"Filtering assets older than {years} years ({cutoff})")

    recent, skipped = [], 0
    for asset in assets:
        created_at_str = asset.get("fileCreatedAt")
        if not created_at_str:
            continue

        try:
            # Handle ISO8601 with 'Z' suffix
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            if created_at > cutoff:
                recent.append(asset)
            else:
                skipped += 1
        except ValueError:
            continue

    logger.info(f"Retained {len(recent)} assets (filtered {skipped} old assets).")
    return recent
