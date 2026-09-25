"""Settings from the environment, `.env`, and the saved Immich connection."""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

CONNECTION_FILE = Path(".immich_config.json")
SECRETS = ("IMMICH_URL", "API_KEY", "FRIGATE_URL", "FRIGATE_USER", "FRIGATE_PASSWORD")


@dataclass(frozen=True)
class Settings:
    IMMICH_URL: str = ""
    API_KEY: str = ""
    OUTPUT_DIR: str = "frigate_train"
    CACHE_DIR: str = ".if_cache"
    YEARS_FILTER: int = 10
    MAX_IMAGES: int = 30
    MIN_FACE_SIZE: int = 80
    BLUR_THRESHOLD: float = 50.0
    REJECT_GRAYSCALE: bool = True
    USE_FULL_RESOLUTION: bool = True
    FRIGATE_RECOGNITION_THRESHOLD: float = 0.9
    FORCE_CPU: bool = False
    FRIGATE_URL: str = ""
    FRIGATE_USER: str = ""
    FRIGATE_PASSWORD: str = ""

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in SECRETS}


def _parse(name: str, default, raw: str):
    if isinstance(default, bool):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be true or false, not {raw!r}")
    try:
        return type(default)(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be a {'whole ' if isinstance(default, int) else ''}number, not {raw!r}"
        ) from None


def load_settings(env=None) -> Settings:
    """Environment variables win over `.env`, which wins over the saved connection."""
    if env is None:
        load_dotenv()
        env = os.environ
    saved = json.loads(CONNECTION_FILE.read_text()) if CONNECTION_FILE.exists() else {}
    values = {}
    for field in fields(Settings):
        raw = env.get(field.name) or saved.get(field.name)
        if raw not in (None, ""):
            values[field.name] = _parse(field.name, field.default, str(raw))
    settings = Settings(**values)
    for name in ("YEARS_FILTER", "MAX_IMAGES", "MIN_FACE_SIZE"):
        if getattr(settings, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    if settings.BLUR_THRESHOLD < 0:
        raise ValueError("BLUR_THRESHOLD must not be negative")
    if not 0 < settings.FRIGATE_RECOGNITION_THRESHOLD < 1:
        raise ValueError("FRIGATE_RECOGNITION_THRESHOLD must be between 0 and 1")
    return settings


def save_connection(url: str, api_key: str) -> None:
    """Store the connection readable only by the current user; it contains an API key."""
    CONNECTION_FILE.touch(mode=0o600, exist_ok=True)
    CONNECTION_FILE.chmod(0o600)
    CONNECTION_FILE.write_text(json.dumps({"IMMICH_URL": url, "API_KEY": api_key}, indent=2) + "\n")
