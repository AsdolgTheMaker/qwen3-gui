"""
Dataset builder tab widget.
"""

import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

from ..constants import DATASETS_DIR


class DatasetBuilderTab(QWidget):
    """Interface for building voice training datasets."""

    def __init__(self):
        super().__init__()
        self.dataset_entries = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel(
            "<b>Dataset Builder</b><br>"
            "Build a dataset for voice training by matching audio files with their transcripts.<br>"
            "You can import audio files and manually transcribe them, or import existing transcript files."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Import buttons
        import_layout = QHBoxLayout()

        import_audio_btn = QPushButton("Import Audio Files...")
        import_audio_btn.clicked.connect(self._import_audio)
        import_layout.addWidget(import_audio_btn)

        import_transcript_btn = QPushButton("Import Transcript File...")
        import_transcript_btn.clicked.connect(self._import_transcript)
        import_layout.addWidget(import_transcript_btn)

        import_folder_btn = QPushButton("Import Audio Folder...")
        import_folder_btn.clicked.connect(self._import_folder)
        import_layout.addWidget(import_folder_btn)

        import_layout.addStretch()
        layout.addLayout(import_layout)

        # Dataset table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Audio File", "Duration", "Transcript", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        # Dataset info
        self.info_label = QLabel("Dataset: 0 entries, 0:00 total duration")
        layout.addWidget(self.info_label)

        # Save options
        save_layout = QHBoxLayout()

        save_layout.addWidget(QLabel("Dataset Name:"))
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("my_voice_dataset")
        self.dataset_name_edit.setMaximumWidth(200)
        save_layout.addWidget(self.dataset_name_edit)

        save_btn = QPushButton("Save Dataset")
        save_btn.clicked.connect(self._save_dataset)
        save_layout.addWidget(save_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        save_layout.addWidget(clear_btn)

        save_layout.addStretch()
        layout.addLayout(save_layout)

    def _import_audio(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files",
            "", "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*.*)"
        )
        for f in files:
            self._add_entry(f, "")

    def _import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Audio Folder")
        if folder:
            folder_path = Path(folder)
            audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
            for f in sorted(folder_path.iterdir()):
                if f.suffix.lower() in audio_extensions:
                    self._add_entry(str(f), "")

    def _import_transcript(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Transcript File",
            "", "Text Files (*.txt *.csv);;All Files (*.*)"
        )
        if file:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "|" in line:
                            audio_name, transcript = line.split("|", 1)
                            # Try to find the audio file
                            audio_path = Path(file).parent / audio_name.strip()
                            if audio_path.exists():
                                self._add_entry(str(audio_path), transcript.strip())
            except Exception as e:
                QMessageBox.warning(self, "Import Error", f"Failed to import transcript: {e}")

    def _add_entry(self, audio_path: str, transcript: str):
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Audio file
        audio_item = QTableWidgetItem(Path(audio_path).name)
        audio_item.setData(Qt.UserRole, audio_path)
        audio_item.setFlags(audio_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 0, audio_item)

        # Duration
        duration = self._get_audio_duration(audio_path)
        duration_item = QTableWidgetItem(duration)
        duration_item.setFlags(duration_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 1, duration_item)

        # Transcript (editable)
        transcript_item = QTableWidgetItem(transcript)
        self.table.setItem(row, 2, transcript_item)

        # Actions
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(4, 0, 4, 0)

        play_btn = QPushButton("Play")
        play_btn.setMaximumWidth(50)
        play_btn.clicked.connect(lambda checked, r=row: self._play_row(r))
        actions_layout.addWidget(play_btn)

        del_btn = QPushButton("X")
        del_btn.setMaximumWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._delete_row(r))
        actions_layout.addWidget(del_btn)

        self.table.setCellWidget(row, 3, actions_widget)

        self._update_info()

    def _get_audio_duration(self, path: str) -> str:
        try:
            import librosa
            duration = librosa.get_duration(path=path)
            mins, secs = divmod(int(duration), 60)
            return f"{mins}:{secs:02d}"
        except Exception:
            return "??:??"

    def _play_row(self, row: int):
        item = self.table.item(row, 0)
        if item:
            audio_path = item.data(Qt.UserRole)
            # Find the main window's media player
            main_window = self.window()
            if hasattr(main_window, 'media_player'):
                main_window.media_player.load_file(audio_path)
                main_window.media_player.player.play()

    def _delete_row(self, row: int):
        self.table.removeRow(row)
        self._update_info()

    def _update_info(self):
        count = self.table.rowCount()
        total_duration = 0
        for row in range(count):
            item = self.table.item(row, 1)
            if item:
                try:
                    parts = item.text().split(":")
                    total_duration += int(parts[0]) * 60 + int(parts[1])
                except Exception:
                    pass
        mins, secs = divmod(total_duration, 60)
        self.info_label.setText(f"Dataset: {count} entries, {mins}:{secs:02d} total duration")

    def _save_dataset(self):
        name = self.dataset_name_edit.text().strip() or "dataset"
        dataset_path = DATASETS_DIR / name
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Save transcript file
        transcript_file = dataset_path / "transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            for row in range(self.table.rowCount()):
                audio_item = self.table.item(row, 0)
                transcript_item = self.table.item(row, 2)
                if audio_item and transcript_item:
                    audio_path = audio_item.data(Qt.UserRole)
                    transcript = transcript_item.text()
                    # Copy audio file to dataset folder
                    dest_audio = dataset_path / Path(audio_path).name
                    if not dest_audio.exists():
                        shutil.copy2(audio_path, dest_audio)
                    f.write(f"{dest_audio.name}|{transcript}\n")

        QMessageBox.information(self, "Dataset Saved", f"Dataset saved to:\n{dataset_path}")

    def _clear_all(self):
        reply = QMessageBox.question(
            self, "Clear Dataset",
            "Are you sure you want to clear all entries?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.table.setRowCount(0)
            self._update_info()
