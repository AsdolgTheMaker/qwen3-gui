"""
Widget components for Qwen3-TTS GUI.
"""

from .media_player import MediaPlayerWidget
from .tts_tab import TTSTab
from .dataset_tab import DatasetBuilderTab
from .training_tab import TrainingTab

__all__ = [
    "MediaPlayerWidget",
    "TTSTab",
    "DatasetBuilderTab",
    "TrainingTab",
]
