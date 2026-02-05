"""
Constants and configuration for Qwen3-TTS GUI.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.parent.absolute()
CONFIG_FILE = SCRIPT_DIR / "config.json"
DATASETS_DIR = SCRIPT_DIR / "datasets"
MODELS_DIR = SCRIPT_DIR / "trained_models"
OUTPUT_DIR = SCRIPT_DIR / "output"

# ---------------------------------------------------------------------------
# Application info
# ---------------------------------------------------------------------------

APP_VERSION = "1.5.22"
GITHUB_REPO = "AsdolgTheMaker/qwen3-gui"
UPDATE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODELS = {
    "Custom Voice (1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Custom Voice (0.6B)": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Voice Design (1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Voice Clone (1.7B)":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Voice Clone (0.6B)":  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

SPEAKER_INFO = {
    "Vivian":   "Bright, slightly edgy young female (Chinese)",
    "Serena":   "Warm, gentle young female (Chinese)",
    "Uncle_Fu": "Seasoned male, low mellow timbre (Chinese)",
    "Dylan":    "Youthful Beijing male, clear natural (Chinese - Beijing)",
    "Eric":     "Lively Chengdu male, slightly husky (Chinese - Sichuan)",
    "Ryan":     "Dynamic male, strong rhythmic drive (English)",
    "Aiden":    "Sunny American male, clear midrange (English)",
    "Ono_Anna": "Playful female, light nimble timbre (Japanese)",
    "Sohee":    "Warm female, rich emotion (Korean)",
}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

DTYPE_OPTIONS = ["bfloat16", "float16", "float32"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def mode_of(label: str) -> str:
    """Determine mode from model label."""
    if "Custom Voice" in label:
        return "custom"
    if "Voice Design" in label:
        return "design"
    return "clone"
