"""
Built-in media player widget for audio playback.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from ..translations import tr


class MediaPlayerWidget(QFrame):
    """Built-in media player for testing generated audio."""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._current_file = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel(tr("audio_player"))
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(title)

        # File label
        self.file_label = QLabel(tr("no_file_loaded"))
        self.file_label.setStyleSheet("color: #666;")
        layout.addWidget(self.file_label)

        # Player setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)

        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self._seek)
        layout.addWidget(self.progress_slider)

        # Time labels
        time_layout = QHBoxLayout()
        self.time_current = QLabel("0:00")
        self.time_total = QLabel("0:00")
        time_layout.addWidget(self.time_current)
        time_layout.addStretch()
        time_layout.addWidget(self.time_total)
        layout.addLayout(time_layout)

        # Control buttons
        btn_layout = QHBoxLayout()

        self.btn_play = QPushButton(tr("play"))
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        btn_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton(tr("stop"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()

        # Volume
        btn_layout.addWidget(QLabel(tr("vol")))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self._set_volume)
        btn_layout.addWidget(self.volume_slider)

        layout.addLayout(btn_layout)

        # Connect signals
        self.player.positionChanged.connect(self._update_position)
        self.player.durationChanged.connect(self._update_duration)
        self.player.playbackStateChanged.connect(self._state_changed)

    def load_file(self, path: str):
        """Load an audio file for playback."""
        self._current_file = path
        self.file_label.setText(Path(path).name)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress_slider.setEnabled(True)

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop(self):
        self.player.stop()

    def _seek(self, position: int):
        self.player.setPosition(position)

    def _set_volume(self, value: int):
        self.audio_output.setVolume(value / 100.0)

    def _update_position(self, position: int):
        self.progress_slider.setValue(position)
        mins, secs = divmod(position // 1000, 60)
        self.time_current.setText(f"{mins}:{secs:02d}")

    def _update_duration(self, duration: int):
        self.progress_slider.setRange(0, duration)
        mins, secs = divmod(duration // 1000, 60)
        self.time_total.setText(f"{mins}:{secs:02d}")

    def _state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setText(tr("pause"))
        else:
            self.btn_play.setText(tr("play"))
