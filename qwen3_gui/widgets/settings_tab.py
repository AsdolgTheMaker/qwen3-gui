"""
Settings tab for application configuration.
"""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QCheckBox, QMessageBox, QComboBox
)
from PySide6.QtGui import QFont

from ..translations import tr
from ..settings import (
    get_auto_update_enabled, set_auto_update_enabled,
    get_hf_cache_path, set_hf_cache_path, apply_hf_cache_env,
    get_whisper_model, set_whisper_model
)

# Available Whisper models with descriptions
WHISPER_MODELS = {
    "openai/whisper-tiny": "Tiny (~39M) - Fastest, lowest accuracy",
    "openai/whisper-base": "Base (~74M) - Fast, good accuracy",
    "openai/whisper-small": "Small (~244M) - Balanced",
    "openai/whisper-medium": "Medium (~769M) - High accuracy",
    "openai/whisper-large-v3": "Large V3 (~1.5B) - Best accuracy, slowest",
}


class SettingsTab(QWidget):
    """Settings configuration tab."""

    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # HuggingFace Cache Settings
        hf_group = QGroupBox(tr("hf_cache_settings"))
        hf_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        hf_layout = QVBoxLayout(hf_group)

        # Description
        desc = QLabel(tr("hf_cache_description"))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666;")
        hf_layout.addWidget(desc)

        # Path input
        path_layout = QHBoxLayout()
        self.hf_cache_edit = QLineEdit()
        path_layout.addWidget(self.hf_cache_edit)

        browse_btn = QPushButton(tr("browse"))
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_hf_cache)
        path_layout.addWidget(browse_btn)

        hf_layout.addLayout(path_layout)

        # Note about restart
        note = QLabel(tr("hf_cache_note"))
        note.setStyleSheet("color: #888; font-style: italic;")
        hf_layout.addWidget(note)

        layout.addWidget(hf_group)

        # Whisper Model Settings
        whisper_group = QGroupBox(tr("whisper_settings"))
        whisper_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        whisper_layout = QVBoxLayout(whisper_group)

        whisper_desc = QLabel(tr("whisper_description"))
        whisper_desc.setWordWrap(True)
        whisper_desc.setStyleSheet("color: #666;")
        whisper_layout.addWidget(whisper_desc)

        whisper_select_layout = QHBoxLayout()
        whisper_select_layout.addWidget(QLabel(tr("whisper_model")))
        self.whisper_combo = QComboBox()
        for model_id, desc in WHISPER_MODELS.items():
            self.whisper_combo.addItem(desc, model_id)
        whisper_select_layout.addWidget(self.whisper_combo, stretch=1)
        whisper_layout.addLayout(whisper_select_layout)

        whisper_note = QLabel(tr("whisper_note"))
        whisper_note.setStyleSheet("color: #888; font-style: italic;")
        whisper_layout.addWidget(whisper_note)

        layout.addWidget(whisper_group)

        # Updates Settings
        update_group = QGroupBox(tr("update_settings"))
        update_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        update_layout = QVBoxLayout(update_group)

        self.auto_update_check = QCheckBox(tr("auto_update_on_startup"))
        update_layout.addWidget(self.auto_update_check)

        layout.addWidget(update_group)

        # Save button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton(tr("save_settings"))
        save_btn.setMinimumWidth(150)
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def _load_settings(self):
        """Load current settings into UI."""
        # Get current HF cache path: saved setting > env var > default
        hf_path = get_hf_cache_path()
        if not hf_path:
            hf_path = os.environ.get("HF_HOME", "")
        if not hf_path:
            # Show HuggingFace default cache location
            hf_path = str(Path.home() / ".cache" / "huggingface")
        self.hf_cache_edit.setText(hf_path)

        # Whisper model
        whisper_model = get_whisper_model()
        idx = self.whisper_combo.findData(whisper_model)
        if idx >= 0:
            self.whisper_combo.setCurrentIndex(idx)

        self.auto_update_check.setChecked(get_auto_update_enabled())

    def _browse_hf_cache(self):
        """Browse for HuggingFace cache directory."""
        current = self.hf_cache_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, tr("select_hf_cache_dir"), current
        )
        if path:
            self.hf_cache_edit.setText(path)

    def _save_settings(self):
        """Save settings and apply immediately."""
        hf_path = self.hf_cache_edit.text().strip()
        set_hf_cache_path(hf_path)
        apply_hf_cache_env(hf_path)

        # Save Whisper model (clears cached model if changed)
        new_whisper = self.whisper_combo.currentData()
        old_whisper = get_whisper_model()
        set_whisper_model(new_whisper)

        # Clear cached Whisper model if changed
        if new_whisper != old_whisper:
            from ..workers import TranscriptionWorker
            TranscriptionWorker._model_holder = {"model": None, "processor": None}

        set_auto_update_enabled(self.auto_update_check.isChecked())

        QMessageBox.information(
            self,
            tr("settings_saved"),
            tr("settings_saved_msg")
        )
