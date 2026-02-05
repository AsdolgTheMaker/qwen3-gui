"""
Main application window.
"""

import sys
import os
import subprocess
import webbrowser

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QSplitter, QMessageBox, QApplication, QComboBox, QLabel
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence

from .constants import APP_VERSION, GITHUB_REPO, UPDATE_URL, OUTPUT_DIR
from .translations import tr, LANGUAGES, get_language, set_language
from .widgets import MediaPlayerWidget, OutputLogWidget, TTSTab, DatasetBuilderTab, TrainingTab


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Load saved language preference
        settings = QSettings("AsdolgTheMaker", "Qwen3TTS")
        saved_lang = settings.value("language", "en")
        set_language(saved_lang)

        self.setWindowTitle(f"{tr('app_title')} v{APP_VERSION}")
        self._setup_ui()
        self._setup_menu()
        self._restore_geometry()

    def _setup_ui(self):
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left side - tabs
        self.tabs = QTabWidget()

        # Shared widgets
        self.media_player = MediaPlayerWidget()
        self.output_log = OutputLogWidget()

        # TTS tab (with access to media player and log)
        self.tts_tab = TTSTab(self.media_player, self.output_log)
        self.tabs.addTab(self.tts_tab, tr("tab_tts"))

        # Dataset builder tab
        self.dataset_tab = DatasetBuilderTab()
        self.tabs.addTab(self.dataset_tab, tr("tab_dataset"))

        # Training tab (with access to log)
        self.training_tab = TrainingTab(self.output_log)
        self.tabs.addTab(self.training_tab, tr("tab_training"))

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

        content_layout.addWidget(splitter)
        main_layout.addLayout(content_layout, stretch=1)

        # Bottom bar with language selector
        bottom_bar = QWidget()
        bottom_bar.setMaximumHeight(35)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(10, 2, 10, 5)

        bottom_layout.addStretch()

        # Language selector
        lang_label = QLabel(tr("interface_language"))
        bottom_layout.addWidget(lang_label)

        self.lang_combo = QComboBox()
        self.lang_combo.setMaximumWidth(120)
        for code, name in LANGUAGES.items():
            self.lang_combo.addItem(name, code)

        # Set current language
        current_lang = get_language()
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == current_lang:
                self.lang_combo.setCurrentIndex(i)
                break

        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        bottom_layout.addWidget(self.lang_combo)

        main_layout.addWidget(bottom_bar)

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        self.file_menu = menubar.addMenu(tr("menu_file"))

        self.open_output_action = QAction(tr("menu_open_output"), self)
        self.open_output_action.triggered.connect(self._open_output_folder)
        self.file_menu.addAction(self.open_output_action)

        self.file_menu.addSeparator()

        self.exit_action = QAction(tr("menu_exit"), self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # Help menu
        self.help_menu = menubar.addMenu(tr("menu_help"))

        self.check_update_action = QAction(tr("menu_check_updates"), self)
        self.check_update_action.triggered.connect(self._check_updates)
        self.help_menu.addAction(self.check_update_action)

        self.help_menu.addSeparator()

        self.about_action = QAction(tr("menu_about"), self)
        self.about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(self.about_action)

    def _on_language_changed(self, index: int):
        lang_code = self.lang_combo.itemData(index)
        set_language(lang_code)

        # Save preference
        settings = QSettings("AsdolgTheMaker", "Qwen3TTS")
        settings.setValue("language", lang_code)

        # Show restart message
        if lang_code == "ru":
            msg = "Язык изменён. Перезапустите приложение для применения изменений."
        else:
            msg = "Language changed. Please restart the application to apply changes."

        QMessageBox.information(self, "Qwen3-TTS", msg)

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
                        tr("update_available"),
                        tr("update_message", version=latest_version, current=APP_VERSION),
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        webbrowser.open(data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases"))
                else:
                    QMessageBox.information(self, tr("up_to_date"), tr("up_to_date_msg"))
            else:
                QMessageBox.warning(self, tr("update_failed"), tr("update_failed_msg"))
        except Exception as e:
            QMessageBox.warning(self, tr("update_failed"), f"{tr('update_failed_msg')}\n{e}")

    def _show_about(self):
        QMessageBox.about(
            self,
            tr("about_title"),
            f"<h2>{tr('app_title')}</h2>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>{tr('about_description')}</p>"
            f"<p>{tr('about_features')}</p>"
            f"<ul>"
            f"<li>{tr('about_feature_1')}</li>"
            f"<li>{tr('about_feature_2')}</li>"
            f"<li>{tr('about_feature_3')}</li>"
            f"<li>{tr('about_feature_4')}</li>"
            f"</ul>"
            f"<p><a href='https://github.com/{GITHUB_REPO}'>GitHub Repository</a></p>"
        )
