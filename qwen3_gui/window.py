"""
Main application window.
"""

import sys
import os
import subprocess
import webbrowser

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QSplitter, QMessageBox, QApplication
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence

from .constants import APP_VERSION, GITHUB_REPO, UPDATE_URL, OUTPUT_DIR
from .widgets import MediaPlayerWidget, OutputLogWidget, TTSTab, DatasetBuilderTab, TrainingTab


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Qwen3-TTS GUI v{APP_VERSION}")
        self._setup_ui()
        self._setup_menu()
        self._restore_geometry()

    def _setup_ui(self):
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left side - tabs
        self.tabs = QTabWidget()

        # Shared widgets
        self.media_player = MediaPlayerWidget()
        self.output_log = OutputLogWidget()

        # TTS tab (with access to media player and log)
        self.tts_tab = TTSTab(self.media_player, self.output_log)
        self.tabs.addTab(self.tts_tab, "Text-to-Speech")

        # Dataset builder tab
        self.dataset_tab = DatasetBuilderTab()
        self.tabs.addTab(self.dataset_tab, "Dataset Builder")

        # Training tab (with access to log)
        self.training_tab = TrainingTab(self.output_log)
        self.tabs.addTab(self.training_tab, "Voice Training")

        splitter.addWidget(self.tabs)

        # Right side - media player and output log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # Media player at top
        right_layout.addWidget(self.media_player)

        # Output log takes remaining space
        right_layout.addWidget(self.output_log, stretch=1)

        splitter.addWidget(right_panel)

        # Set splitter sizes (65% tabs, 35% right panel)
        splitter.setSizes([650, 350])

        main_layout.addWidget(splitter)

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_output_action = QAction("Open Output Folder", self)
        open_output_action.triggered.connect(self._open_output_folder)
        file_menu.addAction(open_output_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        check_update_action = QAction("Check for Updates", self)
        check_update_action.triggered.connect(self._check_updates)
        help_menu.addAction(check_update_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _restore_geometry(self):
        settings = QSettings("AsdolgTheMaker", "Qwen3TTS")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Default to near-fullscreen
            screen = QApplication.primaryScreen().availableGeometry()
            self.setGeometry(
                screen.x() + 50,
                screen.y() + 50,
                screen.width() - 100,
                screen.height() - 100
            )

    def closeEvent(self, event):
        settings = QSettings("AsdolgTheMaker", "Qwen3TTS")
        settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    def _open_output_folder(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(OUTPUT_DIR)
        elif sys.platform == "darwin":
            subprocess.run(["open", OUTPUT_DIR])
        else:
            subprocess.run(["xdg-open", OUTPUT_DIR])

    def _check_updates(self):
        try:
            import requests
            response = requests.get(UPDATE_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get("tag_name", "").lstrip("v")
                if latest_version and latest_version != APP_VERSION:
                    reply = QMessageBox.question(
                        self,
                        "Update Available",
                        f"A new version ({latest_version}) is available!\n"
                        f"Current version: {APP_VERSION}\n\n"
                        "Would you like to open the download page?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        webbrowser.open(data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases"))
                else:
                    QMessageBox.information(self, "Up to Date", "You have the latest version!")
            else:
                QMessageBox.warning(self, "Update Check Failed", "Could not check for updates.")
        except Exception as e:
            QMessageBox.warning(self, "Update Check Failed", f"Error checking for updates: {e}")

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Qwen3-TTS GUI",
            f"<h2>Qwen3-TTS GUI</h2>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>A modern interface for Qwen3 Text-to-Speech.</p>"
            f"<p>Features:</p>"
            f"<ul>"
            f"<li>Multiple voice modes (Custom, Design, Clone)</li>"
            f"<li>Built-in media player</li>"
            f"<li>Dataset builder</li>"
            f"<li>Voice training tools</li>"
            f"</ul>"
            f"<p><a href='https://github.com/{GITHUB_REPO}'>GitHub Repository</a></p>"
        )
