"""
Application settings management.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.absolute()
SETTINGS_FILE = SCRIPT_DIR / ".settings.json"


def _load_settings() -> dict:
    """Load settings from file."""
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(data: dict):
    """Save settings to file."""
    try:
        existing = _load_settings()
        existing.update(data)
        SETTINGS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_auto_update_enabled() -> bool:
    """Get auto-update setting."""
    return _load_settings().get("auto_update", True)


def set_auto_update_enabled(enabled: bool):
    """Save auto-update setting."""
    _save_settings({"auto_update": enabled})


def get_hf_cache_path() -> str:
    """Get custom HuggingFace cache path (empty = use system default)."""
    return _load_settings().get("hf_cache_path", "")


def set_hf_cache_path(path: str):
    """Save custom HuggingFace cache path."""
    _save_settings({"hf_cache_path": path})
