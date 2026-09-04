"""Configuration management for if-curator."""

import json
import logging
import math
import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from rich.prompt import Prompt

load_dotenv()

CONFIG_FILE = Path(".immich_config.json")


class Config:
    """Singleton configuration with uppercase attribute access for backward compatibility."""

    _instance: ClassVar["Config | None"] = None

    # Configuration values
    IMMICH_URL: str | None = None
    API_KEY: str | None = None
    OUTPUT_DIR: str = "./frigate_train"
    YEARS_FILTER: int = 10

    # Quality filtering
    MIN_FACE_WIDTH: int = 100
    BLUR_THRESHOLD: float = 100.0
    MIN_CONFIDENCE: float = 0.7
    MAX_AUTO_IMAGES: int = 80

    # Output quality
    FACE_MARGIN: float = 0.15
    USE_FULL_RESOLUTION: bool = True
    ENABLE_FACE_ALIGNMENT: bool = False

    FACE_MAX_IMAGES: int = 30
    FACE_BURST_SECONDS: float = 2.0
    FACE_PIXEL_DUPLICATE_DISTANCE: float = 0.02
    FACE_IDENTITY_MARGIN: float = 0.1
    FACE_OPTIMIZATION_EPSILON: float = 0.00001
    FRIGATE_VERSION: str = "0.17.2"
    FRIGATE_MODEL_DIR: str = ".if_cache/frigate"
    FRIGATE_UNKNOWN_SCORE: float = 0.8
    FRIGATE_RECOGNITION_THRESHOLD: float = 0.9
    FRIGATE_BLUR_CONFIDENCE_FILTER: bool = True
    CAMERA_MANIFEST: str = ""
    FACE_OUTLIER_MAD: float = 3.0
    REJECT_GRAYSCALE: bool = True
    FORCE_CPU: bool = False

    # Caching (opt-in to avoid unexpected files)
    ENABLE_CACHE: bool = False
    CACHE_DIR: str = ".if_cache"

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load configuration from environment and config file."""
        # Load from environment (highest priority)
        self.IMMICH_URL = os.getenv("IMMICH_URL")
        self.API_KEY = os.getenv("API_KEY")
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./frigate_train")
        for name in self.setting_names():
            default = getattr(type(self), name)
            raw = os.getenv(name)
            if raw is None:
                setattr(self, name, default)
            elif isinstance(default, bool):
                if raw.lower() not in {"true", "false", "1", "0", "yes", "no"}:
                    raise ValueError(f"{name} must be a boolean")
                setattr(self, name, raw.lower() in {"true", "1", "yes"})
            else:
                try:
                    setattr(self, name, type(default)(raw))
                except ValueError as exc:
                    raise ValueError(f"Invalid {name}") from exc
        self.validate_settings()

        # Fall back to config file for missing values
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                self.IMMICH_URL = self.IMMICH_URL or data.get("IMMICH_URL")
                self.API_KEY = self.API_KEY or data.get("API_KEY")
                if not os.getenv("OUTPUT_DIR"):
                    self.OUTPUT_DIR = data.get("OUTPUT_DIR", self.OUTPUT_DIR)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Failed to load config file: {e}")

    @staticmethod
    def setting_names() -> tuple[str, ...]:
        return (
            "YEARS_FILTER",
            "MIN_FACE_WIDTH",
            "BLUR_THRESHOLD",
            "MIN_CONFIDENCE",
            "MAX_AUTO_IMAGES",
            "FACE_MAX_IMAGES",
            "FACE_MARGIN",
            "USE_FULL_RESOLUTION",
            "ENABLE_FACE_ALIGNMENT",
            "ENABLE_CACHE",
            "CACHE_DIR",
            "FORCE_CPU",
            "FACE_BURST_SECONDS",
            "FACE_PIXEL_DUPLICATE_DISTANCE",
            "FACE_OPTIMIZATION_EPSILON",
            "FACE_IDENTITY_MARGIN",
            "FRIGATE_VERSION",
            "FRIGATE_MODEL_DIR",
            "FRIGATE_UNKNOWN_SCORE",
            "FRIGATE_RECOGNITION_THRESHOLD",
            "FRIGATE_BLUR_CONFIDENCE_FILTER",
            "CAMERA_MANIFEST",
            "FACE_OUTLIER_MAD",
            "REJECT_GRAYSCALE",
        )

    def validate_settings(self) -> None:
        for name in ("YEARS_FILTER", "MIN_FACE_WIDTH", "MAX_AUTO_IMAGES", "FACE_MAX_IMAGES"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, lo, hi in (
            ("MIN_CONFIDENCE", 0, 1),
            ("FACE_MARGIN", 0, 1),
            ("FACE_BURST_SECONDS", 0, float("inf")),
            ("FACE_PIXEL_DUPLICATE_DISTANCE", 0, 1),
            ("FACE_OPTIMIZATION_EPSILON", 0, 1),
            ("FACE_IDENTITY_MARGIN", 0, 2),
            ("FRIGATE_UNKNOWN_SCORE", 0, 1),
            ("FRIGATE_RECOGNITION_THRESHOLD", 0, 1),
            ("FACE_OUTLIER_MAD", 0, float("inf")),
            ("BLUR_THRESHOLD", 0, float("inf")),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid {name}: outside allowed range")
        if self.FRIGATE_VERSION != "0.17.2":
            raise ValueError("Only the verified Frigate 0.17.2 large profile is supported")
        if not self.FRIGATE_MODEL_DIR:
            raise ValueError("FRIGATE_MODEL_DIR must not be empty")
        if self.FRIGATE_UNKNOWN_SCORE > self.FRIGATE_RECOGNITION_THRESHOLD:
            raise ValueError("FRIGATE_UNKNOWN_SCORE must not exceed FRIGATE_RECOGNITION_THRESHOLD")
        if not self.CACHE_DIR:
            raise ValueError("CACHE_DIR must not be empty")

    def snapshot(self) -> dict:
        """Public processing settings only; never include credentials."""
        return {name: getattr(self, name) for name in self.setting_names()}

    def save(self) -> None:
        """Persist configuration to file."""
        try:
            CONFIG_FILE.write_text(
                json.dumps(
                    {
                        "IMMICH_URL": self.IMMICH_URL,
                        "API_KEY": self.API_KEY,
                        "OUTPUT_DIR": self.OUTPUT_DIR,
                    },
                    indent=2,
                )
            )
            logging.info(f"Configuration saved to {CONFIG_FILE}")
        except OSError as e:
            logging.error(f"Failed to save config: {e}")

    def interactive_setup(self) -> None:
        """Prompt user for missing configuration."""
        from rich.console import Console

        console = Console()

        if not self.IMMICH_URL:
            console.print("[yellow]Immich URL not found.[/yellow]")
            self.IMMICH_URL = Prompt.ask("Enter Immich URL (e.g. http://192.168.1.5:2283)")
            self.save()

        if not self.API_KEY:
            console.print("[yellow]Immich API Key not found.[/yellow]")
            self.API_KEY = Prompt.ask("Enter Immich API Key", password=True)
            self.save()

    def validate(self) -> None:
        """Raise ValueError if required config is missing."""
        self.validate_settings()
        if not self.IMMICH_URL or not self.API_KEY:
            raise ValueError("Missing Immich URL or API Key.")


# Singleton instance and backward-compatible aliases
Config = Config()  # type: ignore[misc]
ConfigManager = type("ConfigManager", (), {"get": staticmethod(lambda: Config)})


def get_headers() -> dict[str, str]:
    """Return HTTP headers for Immich API requests."""
    return {"x-api-key": Config.API_KEY or "", "Accept": "application/json"}
