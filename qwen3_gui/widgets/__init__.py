"""
Widget components for Qwen3-TTS GUI.
"""

from .media_player import MediaPlayerWidget
from .output_log import OutputLogWidget
from .tts_tab import TTSTab
from .dataset_tab import DatasetBuilderTab
from .training_tab import TrainingTab
from .settings_tab import SettingsTab

__all__ = [
    "MediaPlayerWidget",
    "OutputLogWidget",
    "TTSTab",
    "DatasetBuilderTab",
    "TrainingTab",
    "SettingsTab",
]
