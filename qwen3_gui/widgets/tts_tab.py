"""
Text-to-Speech tab widget.
"""

from pathlib import Path
from typing import Optional

import torch
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QLineEdit, QCheckBox, QProgressBar, QGroupBox,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QScrollArea
)
from PySide6.QtCore import Qt

from ..constants import (
    MODELS, SPEAKERS, SPEAKER_INFO, LANGUAGES, DTYPE_OPTIONS,
    OUTPUT_DIR, mode_of
)
from ..tooltips import set_tooltip
from ..workers import GenerationWorker
from .media_player import MediaPlayerWidget
from .output_log import OutputLogWidget


class TTSTab(QWidget):
    """Main text-to-speech interface."""

    def __init__(self, media_player: MediaPlayerWidget, output_log: OutputLogWidget = None):
        super().__init__()
        self.media_player = media_player
        self.output_log = output_log
        self.model_holder = {"model": None, "model_id": None}
        self.worker = None
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Model selection
        model_group = QGroupBox("Model Selection")
        set_tooltip(model_group, "model")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODELS.keys())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)

        scroll_layout.addWidget(model_group)

        # Language
        lang_group = QGroupBox("Language")
        set_tooltip(lang_group, "language")
        lang_layout = QVBoxLayout(lang_group)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(LANGUAGES)
        lang_layout.addWidget(self.lang_combo)

        scroll_layout.addWidget(lang_group)

        # Speaker (Custom Voice only)
        self.speaker_group = QGroupBox("Speaker")
        set_tooltip(self.speaker_group, "speaker")
        speaker_layout = QVBoxLayout(self.speaker_group)

        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems(SPEAKERS)
        self.speaker_combo.currentTextChanged.connect(self._on_speaker_changed)
        speaker_layout.addWidget(self.speaker_combo)

        self.speaker_desc = QLabel(SPEAKER_INFO["Vivian"])
        self.speaker_desc.setStyleSheet("color: #666; font-style: italic;")
        self.speaker_desc.setWordWrap(True)
        speaker_layout.addWidget(self.speaker_desc)

        scroll_layout.addWidget(self.speaker_group)

        # Text prompt
        text_group = QGroupBox("Text to Speak")
        set_tooltip(text_group, "text_prompt")
        text_layout = QVBoxLayout(text_group)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter the text you want the AI to speak...")
        self.text_edit.setMinimumHeight(100)
        text_layout.addWidget(self.text_edit)

        scroll_layout.addWidget(text_group)

        # Instruction
        self.instruct_group = QGroupBox("Instruction (optional)")
        set_tooltip(self.instruct_group, "instruction")
        instruct_layout = QVBoxLayout(self.instruct_group)

        self.instruct_edit = QTextEdit()
        self.instruct_edit.setPlaceholderText("e.g., 'Speak happily' or 'Sound tired'...")
        self.instruct_edit.setMaximumHeight(80)
        instruct_layout.addWidget(self.instruct_edit)

        scroll_layout.addWidget(self.instruct_group)

        # Reference audio (Clone only)
        self.ref_group = QGroupBox("Reference Audio (for cloning)")
        set_tooltip(self.ref_group, "ref_audio")
        ref_layout = QVBoxLayout(self.ref_group)

        ref_file_layout = QHBoxLayout()
        self.ref_path_edit = QLineEdit()
        self.ref_path_edit.setPlaceholderText("Path to reference audio file...")
        ref_file_layout.addWidget(self.ref_path_edit)

        self.ref_browse_btn = QPushButton("Browse...")
        self.ref_browse_btn.clicked.connect(self._browse_ref)
        ref_file_layout.addWidget(self.ref_browse_btn)

        ref_layout.addLayout(ref_file_layout)

        ref_text_label = QLabel("Reference Transcript:")
        set_tooltip(ref_text_label, "ref_text")
        ref_layout.addWidget(ref_text_label)

        self.ref_text_edit = QTextEdit()
        self.ref_text_edit.setPlaceholderText("What's being said in the reference audio...")
        self.ref_text_edit.setMaximumHeight(60)
        ref_layout.addWidget(self.ref_text_edit)

        self.xvector_check = QCheckBox("X-vector only mode (no transcript needed)")
        set_tooltip(self.xvector_check, "xvector")
        ref_layout.addWidget(self.xvector_check)

        scroll_layout.addWidget(self.ref_group)

        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QGridLayout(advanced_group)
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)

        row = 0

        # Temperature
        temp_label = QLabel("Temperature:")
        set_tooltip(temp_label, "temperature")
        advanced_layout.addWidget(temp_label, row, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.05, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(0.9)
        advanced_layout.addWidget(self.temp_spin, row, 1)
        row += 1

        # Top-K
        topk_label = QLabel("Top-K:")
        set_tooltip(topk_label, "top_k")
        advanced_layout.addWidget(topk_label, row, 0)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 200)
        self.topk_spin.setValue(50)
        advanced_layout.addWidget(self.topk_spin, row, 1)
        row += 1

        # Top-P
        topp_label = QLabel("Top-P:")
        set_tooltip(topp_label, "top_p")
        advanced_layout.addWidget(topp_label, row, 0)
        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.0, 1.0)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setValue(1.0)
        advanced_layout.addWidget(self.topp_spin, row, 1)
        row += 1

        # Repetition penalty
        rep_label = QLabel("Repetition Penalty:")
        set_tooltip(rep_label, "rep_penalty")
        advanced_layout.addWidget(rep_label, row, 0)
        self.rep_spin = QDoubleSpinBox()
        self.rep_spin.setRange(1.0, 2.0)
        self.rep_spin.setSingleStep(0.05)
        self.rep_spin.setValue(1.05)
        advanced_layout.addWidget(self.rep_spin, row, 1)
        row += 1

        # Max tokens
        maxtok_label = QLabel("Max Tokens:")
        set_tooltip(maxtok_label, "max_tokens")
        advanced_layout.addWidget(maxtok_label, row, 0)
        self.maxtok_spin = QSpinBox()
        self.maxtok_spin.setRange(256, 8192)
        self.maxtok_spin.setSingleStep(256)
        self.maxtok_spin.setValue(2048)
        advanced_layout.addWidget(self.maxtok_spin, row, 1)
        row += 1

        # Dtype
        dtype_label = QLabel("Dtype:")
        set_tooltip(dtype_label, "dtype")
        advanced_layout.addWidget(dtype_label, row, 0)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(DTYPE_OPTIONS)
        advanced_layout.addWidget(self.dtype_combo, row, 1)
        row += 1

        # Flash attention
        self.flash_check = QCheckBox("Use Flash Attention 2")
        set_tooltip(self.flash_check, "flash_attn")
        self.flash_check.setChecked(True)
        advanced_layout.addWidget(self.flash_check, row, 0, 1, 2)

        scroll_layout.addWidget(advanced_group)

        # Output file
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        output_file_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(str(OUTPUT_DIR / "output.wav"))
        output_file_layout.addWidget(self.output_path_edit)

        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output)
        output_file_layout.addWidget(output_browse_btn)

        output_layout.addLayout(output_file_layout)

        scroll_layout.addWidget(output_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        # Generate button and status at bottom of left panel
        btn_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c3aed;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #6d28d9;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
            }
        """)
        self.generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        left_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel(f"Ready. Device: {self._device}")
        self.status_label.setStyleSheet("color: #666;")
        left_layout.addWidget(self.status_label)

        layout.addWidget(left_panel, stretch=2)

        # Initial mode update
        self._on_model_changed(self.model_combo.currentText())

    def _on_model_changed(self, model_label: str):
        mode = mode_of(model_label)

        # Speaker - only for custom voice
        self.speaker_group.setVisible(mode == "custom")

        # Instruction - for custom and design
        self.instruct_group.setVisible(mode in ("custom", "design"))
        if mode == "design":
            self.instruct_group.setTitle("Voice Description (required)")
        else:
            self.instruct_group.setTitle("Instruction (optional)")

        # Reference - only for clone
        self.ref_group.setVisible(mode == "clone")

    def _on_speaker_changed(self, speaker: str):
        self.speaker_desc.setText(SPEAKER_INFO.get(speaker, ""))

    def _browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio",
            "", "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*.*)"
        )
        if path:
            self.ref_path_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output As",
            self.output_path_edit.text(),
            "WAV Files (*.wav)"
        )
        if path:
            self.output_path_edit.setText(path)

    def _validate(self) -> Optional[str]:
        text = self.text_edit.toPlainText().strip()
        if not text:
            return "Please enter some text to speak."

        mode = mode_of(self.model_combo.currentText())

        if mode == "design":
            instruct = self.instruct_edit.toPlainText().strip()
            if not instruct:
                return "Voice description is required for Voice Design mode."

        if mode == "clone":
            ref = self.ref_path_edit.text().strip()
            if not ref:
                return "Reference audio file is required for Voice Clone mode."
            if not Path(ref).is_file():
                return f"Reference audio file not found: {ref}"
            if not self.xvector_check.isChecked():
                ref_text = self.ref_text_edit.toPlainText().strip()
                if not ref_text:
                    return "Reference transcript is required (or enable X-vector only mode)."

        out = self.output_path_edit.text().strip()
        if not out:
            return "Please specify an output file path."

        return None

    def _log(self, method: str, message: str):
        """Helper to log if output_log is available."""
        if self.output_log:
            getattr(self.output_log, method)(message)

    def _on_generate(self):
        error = self._validate()
        if error:
            QMessageBox.warning(self, "Validation Error", error)
            self._log("log_error", f"Validation failed: {error}")
            return

        # Ensure output directory exists
        out_path = Path(self.output_path_edit.text())
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model_label = self.model_combo.currentText()
        text_preview = self.text_edit.toPlainText().strip()[:50]
        if len(self.text_edit.toPlainText().strip()) > 50:
            text_preview += "..."

        self._log("log_info", f"Starting generation with {model_label}")
        self._log("log", f"Text: \"{text_preview}\"")
        self._log("log", f"Language: {self.lang_combo.currentText()}, Speaker: {self.speaker_combo.currentText()}")

        params = {
            "model_label": model_label,
            "language": self.lang_combo.currentText(),
            "speaker": self.speaker_combo.currentText(),
            "text": self.text_edit.toPlainText().strip(),
            "instruction": self.instruct_edit.toPlainText().strip(),
            "ref_audio": self.ref_path_edit.text().strip(),
            "ref_text": self.ref_text_edit.toPlainText().strip(),
            "xvector": self.xvector_check.isChecked(),
            "temperature": self.temp_spin.value(),
            "top_k": self.topk_spin.value(),
            "top_p": self.topp_spin.value(),
            "rep_penalty": self.rep_spin.value(),
            "max_tokens": self.maxtok_spin.value(),
            "dtype": self.dtype_combo.currentText(),
            "flash_attn": self.flash_check.isChecked(),
            "output_path": str(out_path),
            "device": self._device,
        }

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.show()

        self.worker = GenerationWorker(params, self.model_holder)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self._log("log_warning", "Generation cancelled by user")

    def _on_progress(self, message: str):
        self.status_label.setText(message)
        self._log("log_progress", message)

    def _on_finished(self, success: bool, message: str, output_path: str):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.hide()
        self.status_label.setText(message)

        if success and output_path:
            self._log("log_success", f"Generation complete!")
            self._log("log_audio_saved", output_path)
            self.media_player.load_file(output_path)
        elif not success:
            self._log("log_error", message)
