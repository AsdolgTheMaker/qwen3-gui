"""
Application settings management.
"""

import json
import os
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
    """Get HuggingFace cache path from settings."""
    return _load_settings().get("hf_cache_path", "")


def set_hf_cache_path(path: str):
    """Save HuggingFace cache path."""
    _save_settings({"hf_cache_path": path})


def apply_hf_cache_env(path: str):
    """Apply HuggingFace cache path to environment variables."""
    if path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # HF_HOME is the main variable - others derive from it
        os.environ["HF_HOME"] = path

        # If huggingface_hub was already imported, reload its constants
        # to pick up the new env var value
        import sys
        if "huggingface_hub.constants" in sys.modules:
            import importlib
            from huggingface_hub import constants
            importlib.reload(constants)
