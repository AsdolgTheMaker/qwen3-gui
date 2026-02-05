"""
Application entry point and main function.
"""

import sys
import os
import subprocess

# Apply HF cache setting (in case run.py didn't set it, e.g., running main.py directly)
from .settings import get_hf_cache_path, apply_hf_cache_env
apply_hf_cache_env(get_hf_cache_path())


def _ensure_dependencies():
    """Check and install missing dependencies (for users with old run.py)."""
    missing = []

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")

    if missing:
        print(f"[setup] Installing missing dependencies: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *missing
        ])
        print("[setup] Please restart the application.")
        sys.exit(0)

from PySide6.QtWidgets import QApplication, QToolTip, QStyleFactory
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .constants import APP_VERSION, OUTPUT_DIR, DATASETS_DIR, MODELS_DIR
from .window import MainWindow


def main():
    """Main entry point for the Qwen3-TTS GUI application."""

    # Check for missing dependencies (helps users with old run.py)
    _ensure_dependencies()

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Enable High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    # Set application info
    app.setApplicationName("Qwen3-TTS GUI")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("AsdolgTheMaker")

    # Enable rich tooltips
    QToolTip.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
