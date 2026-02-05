"""
Settings tab for application configuration.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ..translations import tr
from ..settings import (
    get_auto_update_enabled, set_auto_update_enabled,
    get_hf_cache_path, set_hf_cache_path
)


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
        self.hf_cache_edit.setPlaceholderText(tr("hf_cache_placeholder"))
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
        self.hf_cache_edit.setText(get_hf_cache_path())
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
        """Save settings."""
        set_hf_cache_path(self.hf_cache_edit.text().strip())
        set_auto_update_enabled(self.auto_update_check.isChecked())

        QMessageBox.information(
            self,
            tr("settings_saved"),
            tr("settings_saved_msg")
        )
