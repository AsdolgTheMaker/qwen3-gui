"""
Built-in media player widget for audio playback.
Uses sounddevice for reliable playback instead of Qt's QMediaPlayer.
"""

import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont

from ..translations import tr

# Try to import sounddevice, fall back gracefully
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
    SOUNDDEVICE_ERROR = None
except Exception as e:
    HAS_SOUNDDEVICE = False
    SOUNDDEVICE_ERROR = str(e)


class MediaPlayerWidget(QFrame):
    """Built-in media player for testing generated audio."""

    # Signal for logging messages to GUI log
    log_signal = Signal(str)
    log_error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._current_file = None
        self._audio_data = None
        self._sample_rate = 0
        self._duration_ms = 0
        self._playing = False
        self._paused = False
        self._position_ms = 0
        self._stream = None
        self._play_thread = None
        self._lock = threading.Lock()

        self._setup_ui()

        # Timer for updating position during playback
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_playback_position)
        self._update_timer.setInterval(50)  # Update every 50ms

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

        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)
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

        self._volume = 0.8
        self._seeking = False

    def load_file(self, path: str):
        """Load an audio file for playback."""
        # Stop any current playback
        self._stop()

        self._current_file = path
        self.file_label.setText(Path(path).name)

        try:
            # Load audio data
            self._audio_data, self._sample_rate = sf.read(path, dtype='float32')

            # Convert mono to stereo if needed
            if len(self._audio_data.shape) == 1:
                self._audio_data = np.column_stack([self._audio_data, self._audio_data])

            # Calculate duration
            self._duration_ms = int(len(self._audio_data) / self._sample_rate * 1000)
            self._position_ms = 0

            # Update UI
            self.progress_slider.setRange(0, self._duration_ms)
            self.progress_slider.setValue(0)
            mins, secs = divmod(self._duration_ms // 1000, 60)
            self.time_total.setText(f"{mins}:{secs:02d}")
            self.time_current.setText("0:00")

            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.progress_slider.setEnabled(True)

            # Show the player (it starts hidden)
            self.show()

            # Log success
            self.log_signal.emit(f"MediaPlayer: Loaded {Path(path).name} ({mins}:{secs:02d}, {self._sample_rate}Hz)")

        except Exception as e:
            self.log_error_signal.emit(f"MediaPlayer: Error loading file: {e}")
            self.file_label.setText(f"Error: {e}")

    def _toggle_play(self):
        if self._playing and not self._paused:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not HAS_SOUNDDEVICE or self._audio_data is None:
            if not HAS_SOUNDDEVICE:
                self.log_error_signal.emit(f"Audio playback unavailable: {SOUNDDEVICE_ERROR}")
            return

        if self._paused:
            # Resume from pause
            self._paused = False
            self.btn_play.setText(tr("pause"))
            return

        # Stop any existing playback first
        if self._playing:
            self._playing = False
            if self._play_thread and self._play_thread.is_alive():
                self._play_thread.join(timeout=0.5)

        # Start new playback
        self._playing = True
        self._paused = False
        self.btn_play.setText(tr("pause"))
        self._update_timer.start()

        # Start playback in a thread
        self._play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._play_thread.start()

    def _playback_worker(self):
        """Background thread for audio playback."""
        try:
            # Calculate starting sample
            start_sample = int(self._position_ms / 1000 * self._sample_rate)
            audio_to_play = self._audio_data[start_sample:]

            # Apply volume
            audio_to_play = audio_to_play * self._volume

            # Track playback position
            samples_played = 0
            block_size = 1024

            def callback(outdata, frames, time_info, status):
                nonlocal samples_played

                if self._paused:
                    outdata.fill(0)
                    return

                with self._lock:
                    current_pos = start_sample + samples_played
                    end_pos = min(current_pos + frames, len(self._audio_data))
                    actual_frames = end_pos - current_pos

                    if actual_frames <= 0:
                        outdata.fill(0)
                        raise sd.CallbackStop()

                    chunk = self._audio_data[current_pos:end_pos] * self._volume

                    if actual_frames < frames:
                        outdata[:actual_frames] = chunk
                        outdata[actual_frames:].fill(0)
                        samples_played += actual_frames
                        raise sd.CallbackStop()
                    else:
                        outdata[:] = chunk
                        samples_played += frames

                    # Update position
                    self._position_ms = int((start_sample + samples_played) / self._sample_rate * 1000)

            with sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._audio_data.shape[1] if len(self._audio_data.shape) > 1 else 1,
                callback=callback,
                blocksize=block_size,
                dtype='float32'
            ) as stream:
                while stream.active and self._playing:
                    sd.sleep(50)

        except Exception as e:
            # Schedule error log on main thread
            QTimer.singleShot(0, lambda: self.log_error_signal.emit(f"MediaPlayer: Playback error: {e}"))
        finally:
            self._playing = False
            self._paused = False
            # Schedule UI update on main thread
            QTimer.singleShot(0, self._on_playback_finished)

    def _on_playback_finished(self):
        """Called when playback finishes."""
        self._update_timer.stop()
        self.btn_play.setText(tr("play"))
        if self._position_ms >= self._duration_ms - 100:  # Near end
            self._position_ms = 0
            self.progress_slider.setValue(0)
            self.time_current.setText("0:00")

    def _pause(self):
        self._paused = True
        self.btn_play.setText(tr("play"))

    def _stop(self):
        self._playing = False
        self._paused = False
        self._position_ms = 0
        self._update_timer.stop()

        # Wait for playback thread to finish
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.5)

        self.btn_play.setText(tr("play"))
        self.progress_slider.setValue(0)
        self.time_current.setText("0:00")

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_released(self):
        self._seeking = False
        # Seek to new position
        new_pos = self.progress_slider.value()
        was_playing = self._playing and not self._paused

        self._stop()
        self._position_ms = new_pos

        mins, secs = divmod(new_pos // 1000, 60)
        self.time_current.setText(f"{mins}:{secs:02d}")
        self.progress_slider.setValue(new_pos)

        if was_playing:
            self._play()

    def _set_volume(self, value: int):
        self._volume = value / 100.0

    def _update_playback_position(self):
        """Update UI during playback."""
        if not self._seeking and self._playing:
            self.progress_slider.setValue(self._position_ms)
            mins, secs = divmod(self._position_ms // 1000, 60)
            self.time_current.setText(f"{mins}:{secs:02d}")
