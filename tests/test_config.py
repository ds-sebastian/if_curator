import json
import stat

import pytest

from if_curator import config
from if_curator.config import Settings, load_settings, save_connection


def test_defaults():
    assert load_settings({}) == Settings()


def test_parses_types():
    settings = load_settings({"MAX_IMAGES": "12", "BLUR_THRESHOLD": "7.5", "FORCE_CPU": "Yes", "REJECT_GRAYSCALE": "0"})
    assert (settings.MAX_IMAGES, settings.BLUR_THRESHOLD, settings.FORCE_CPU, settings.REJECT_GRAYSCALE) == (
        12,
        7.5,
        True,
        False,
    )


@pytest.mark.parametrize(
    "env",
    [
        {"MAX_IMAGES": "many"},
        {"FORCE_CPU": "maybe"},
        {"MIN_FACE_SIZE": "0"},
        {"BLUR_THRESHOLD": "-1"},
        {"FRIGATE_RECOGNITION_THRESHOLD": "1.5"},
    ],
)
def test_rejects_invalid(env):
    with pytest.raises(ValueError):
        load_settings(env)


def test_saved_connection_is_private_and_env_wins():
    save_connection("http://immich:2283", "key")
    assert stat.S_IMODE(config.CONNECTION_FILE.stat().st_mode) == 0o600
    assert json.loads(config.CONNECTION_FILE.read_text()) == {"IMMICH_URL": "http://immich:2283", "API_KEY": "key"}
    assert load_settings({}).API_KEY == "key"
    assert load_settings({"API_KEY": "other"}).API_KEY == "other"


def test_public_settings_hide_credentials():
    public = Settings(IMMICH_URL="http://x", API_KEY="secret").public()
    assert "API_KEY" not in public and "IMMICH_URL" not in public and public["MAX_IMAGES"] == 30
