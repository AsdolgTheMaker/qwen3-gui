"""
Voice training tab widget.
"""

from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QProgressBar, QGroupBox, QSpinBox,
    QDoubleSpinBox, QMessageBox
)
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont

from ..constants import DATASETS_DIR
from ..tooltips import set_tooltip
from ..translations import tr
from ..workers import TrainingWorker
from .output_log import OutputLogWidget


class TrainingTab(QWidget):
    """Interface for training custom voice models."""

    def __init__(self, output_log: OutputLogWidget = None):
        super().__init__()
        self.output_log = output_log
        self._settings = QSettings("AsdolgTheMaker", "Qwen3TTS")
        self._worker = None
        self._setup_ui()
        self._restore_state()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info
        info_label = QLabel(
            f"<b>{tr('training_title')}</b><br>"
            f"{tr('training_info')}"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Dataset selection
        dataset_group = QGroupBox(tr("dataset"))
        dataset_layout = QGridLayout(dataset_group)

        dataset_layout.addWidget(QLabel(tr("select_dataset")), 0, 0)
        self.dataset_combo = QComboBox()
        self._refresh_datasets()
        dataset_layout.addWidget(self.dataset_combo, 0, 1)

        refresh_btn = QPushButton(tr("refresh"))
        refresh_btn.setMaximumWidth(60)
        refresh_btn.clicked.connect(self._refresh_datasets)
        dataset_layout.addWidget(refresh_btn, 0, 2)

        layout.addWidget(dataset_group)

        # Training parameters
        params_group = QGroupBox(tr("training_params"))
        params_layout = QGridLayout(params_group)

        row = 0

        # Base model
        params_layout.addWidget(QLabel(tr("base_model")), row, 0)
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems([
            "Qwen3-TTS-1.7B (recommended)",
            "Qwen3-TTS-0.6B (faster)"
        ])
        params_layout.addWidget(self.base_model_combo, row, 1)
        row += 1

        # Epochs
        epochs_label = QLabel(tr("epochs"))
        set_tooltip(epochs_label, "epochs")
        params_layout.addWidget(epochs_label, row, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin, row, 1)
        row += 1

        # Learning rate
        lr_label = QLabel(tr("learning_rate"))
        set_tooltip(lr_label, "learning_rate")
        params_layout.addWidget(lr_label, row, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0000001, 0.001)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setSingleStep(0.000001)
        self.lr_spin.setValue(0.000002)  # 2e-6, recommended for fine-tuning
        params_layout.addWidget(self.lr_spin, row, 1)
        row += 1

        # Batch size
        batch_label = QLabel(tr("batch_size"))
        set_tooltip(batch_label, "batch_size")
        params_layout.addWidget(batch_label, row, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        params_layout.addWidget(self.batch_spin, row, 1)
        row += 1

        # Output model name
        params_layout.addWidget(QLabel(tr("output_model_name")), row, 0)
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("my_custom_voice")
        params_layout.addWidget(self.model_name_edit, row, 1)

        layout.addWidget(params_group)

        # Training controls
        controls_layout = QHBoxLayout()

        self.train_btn = QPushButton(tr("start_training"))
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
            }
        """)
        self.train_btn.clicked.connect(self._start_training)
        controls_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton(tr("stop_training"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Training log
        log_group = QGroupBox(tr("training_log"))
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

    def _refresh_datasets(self):
        self.dataset_combo.clear()
        if DATASETS_DIR.exists():
            for d in DATASETS_DIR.iterdir():
                if d.is_dir() and (d / "transcript.txt").exists():
                    self.dataset_combo.addItem(d.name)

    def _log(self, method: str, message: str):
        """Helper to log to both local log and output_log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        if self.output_log:
            getattr(self.output_log, method)(message)

    def _start_training(self):
        dataset = self.dataset_combo.currentText()
        if not dataset:
            QMessageBox.warning(self, tr("no_dataset"), tr("no_dataset_msg"))
            return

        model_name = self.model_name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, tr("no_model_name"), tr("no_model_name_msg"))
            return

        self.log_text.clear()
        self._log("log_info", "Starting training...")
        self._log("log", f"Dataset: {dataset}")
        self._log("log", f"Epochs: {self.epochs_spin.value()}")
        self._log("log", f"Learning Rate: {self.lr_spin.value()}")
        self._log("log", f"Batch Size: {self.batch_spin.value()}")

        # Prepare training parameters
        params = {
            "dataset": dataset,
            "model_name": model_name,
            "base_model": self.base_model_combo.currentIndex(),
            "epochs": self.epochs_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "batch_size": self.batch_spin.value(),
            "device": "cuda:0",  # TODO: Make configurable
        }

        # Update UI state
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.show()
        self.progress_bar.setRange(0, params["epochs"])
        self.progress_bar.setValue(0)

        # Create and start worker
        self._worker = TrainingWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.epoch_progress.connect(self._on_epoch_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop_training(self):
        """Stop the training worker."""
        if self._worker:
            self._log("log_warning", "Cancelling training...")
            self._worker.cancel()

    def _on_progress(self, message: str):
        """Handle progress signal from worker."""
        self._log("log_progress", message)

        # Parse download percentage if present (e.g., "Downloading: 50%")
        if "%" in message and "Downloading" in message:
            import re
            match = re.search(r'(\d+)%', message)
            if match:
                pct = int(match.group(1))
                # Switch to 0-100 range for download progress
                if self.progress_bar.maximum() != 100:
                    self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(pct)
        elif "Step 3/3" in message:
            # Reset to epoch range when training starts
            epochs = self.epochs_spin.value()
            self.progress_bar.setRange(0, epochs)
            self.progress_bar.setValue(0)

    def _on_log(self, message: str):
        """Handle log signal from worker."""
        self._log("log", message)

    def _on_epoch_progress(self, epoch: int, total_epochs: int, loss: float):
        """Handle epoch progress signal from worker."""
        self.progress_bar.setValue(epoch)
        self._log("log_info", f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}")

    def _on_finished(self, success: bool, message: str):
        """Handle training finished signal."""
        self._worker = None
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()

        if success:
            self._log("log_success", message)
            QMessageBox.information(self, tr("training_complete"), message)
        else:
            self._log("log_error", message)
            if "cancelled" not in message.lower():
                QMessageBox.warning(self, tr("training_failed"), message)

    def _save_state(self):
        """Save widget state to settings."""
        s = self._settings
        s.beginGroup("training")
        s.setValue("base_model", self.base_model_combo.currentIndex())
        s.setValue("epochs", self.epochs_spin.value())
        s.setValue("learning_rate", self.lr_spin.value())
        s.setValue("batch_size", self.batch_spin.value())
        s.setValue("model_name", self.model_name_edit.text())
        s.endGroup()

    def _restore_state(self):
        """Restore widget state from settings."""
        s = self._settings
        s.beginGroup("training")

        idx = s.value("base_model", 0, type=int)
        if 0 <= idx < self.base_model_combo.count():
            self.base_model_combo.setCurrentIndex(idx)

        self.epochs_spin.setValue(s.value("epochs", 10, type=int))
        self.lr_spin.setValue(s.value("learning_rate", 0.000002, type=float))
        self.batch_spin.setValue(s.value("batch_size", 4, type=int))
        self.model_name_edit.setText(s.value("model_name", "", type=str))
        s.endGroup()

    def hideEvent(self, event):
        """Save state when tab is hidden."""
        self._save_state()
        super().hideEvent(event)
