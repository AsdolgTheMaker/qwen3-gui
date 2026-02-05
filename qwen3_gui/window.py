"""
Main application window.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QSplitter, QMessageBox, QApplication, QMenu
)
from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence

from .constants import APP_VERSION, GITHUB_REPO, OUTPUT_DIR
from .translations import tr, LANGUAGES, get_language, set_language
from .widgets import MediaPlayerWidget, OutputLogWidget, TTSTab, DatasetBuilderTab, TrainingTab, SettingsTab
from .settings import get_auto_update_enabled, set_auto_update_enabled

SCRIPT_DIR = Path(__file__).parent.parent.absolute()


class UpdateWorker(QThread):
    """Background worker for checking/installing updates."""
    finished = Signal(bool, str, str)  # success, message, new_version

    def __init__(self, install: bool = False):
        super().__init__()
        self.install = install

    def run(self):
        try:
            # Import update functions from run.py
            import importlib.util
            spec = importlib.util.spec_from_file_location("run", SCRIPT_DIR / "run.py")
            run_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_module)

            local_ver = run_module._get_local_version()
            remote_ver = run_module._get_remote_version()

            if remote_ver == "0.0.0":
                self.finished.emit(False, "Could not check remote version", "")
                return

            if run_module._version_gt(remote_ver, local_ver):
                if self.install:
                    success = run_module._perform_update()
                    if success:
                        self.finished.emit(True, "update_installed", remote_ver)
                    else:
                        self.finished.emit(False, "Update installation failed", "")
                else:
                    self.finished.emit(True, "update_available", remote_ver)
            else:
                self.finished.emit(True, "up_to_date", local_ver)
        except Exception as e:
            self.finished.emit(False, str(e), "")


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

        self._update_worker = None

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

        # Connect media player log signals to output log
        self.media_player.log_signal.connect(self.output_log.log)
        self.media_player.log_error_signal.connect(self.output_log.log_error)

        # TTS tab (with access to media player and log)
        self.tts_tab = TTSTab(self.media_player, self.output_log)
        self.tabs.addTab(self.tts_tab, tr("tab_tts"))

        # Dataset builder tab
        self.dataset_tab = DatasetBuilderTab()
        self.tabs.addTab(self.dataset_tab, tr("tab_dataset"))

        # Training tab (with access to log)
        self.training_tab = TrainingTab(self.output_log)
        self.tabs.addTab(self.training_tab, tr("tab_training"))

        # Settings tab
        self.settings_tab = SettingsTab()
        self.tabs.addTab(self.settings_tab, tr("tab_settings"))

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

        # Language menu
        self.lang_menu = menubar.addMenu(tr("interface_language").rstrip(":"))
        self.lang_action_group = QActionGroup(self)
        self.lang_action_group.setExclusive(True)

        current_lang = get_language()
        for code, name in LANGUAGES.items():
            action = QAction(name, self, checkable=True)
            action.setData(code)
            action.setChecked(code == current_lang)
            action.triggered.connect(self._on_language_changed)
            self.lang_action_group.addAction(action)
            self.lang_menu.addAction(action)

        # Help menu
        self.help_menu = menubar.addMenu(tr("menu_help"))

        self.check_update_action = QAction(tr("menu_check_updates"), self)
        self.check_update_action.triggered.connect(self._check_updates)
        self.help_menu.addAction(self.check_update_action)

        self.auto_update_action = QAction(tr("menu_auto_update"), self, checkable=True)
        self.auto_update_action.setChecked(get_auto_update_enabled())
        self.auto_update_action.triggered.connect(self._toggle_auto_update)
        self.help_menu.addAction(self.auto_update_action)

        self.help_menu.addSeparator()

        self.about_action = QAction(tr("menu_about"), self)
        self.about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(self.about_action)

    def _on_language_changed(self):
        action = self.lang_action_group.checkedAction()
        if not action:
            return

        lang_code = action.data()
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

    def _toggle_auto_update(self):
        enabled = self.auto_update_action.isChecked()
        set_auto_update_enabled(enabled)

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
        """Check for updates and offer to install."""
        self.check_update_action.setEnabled(False)
        self.check_update_action.setText(tr("checking_updates"))

        self._update_worker = UpdateWorker(install=False)
        self._update_worker.finished.connect(self._on_update_check_done)
        self._update_worker.start()

    def _on_update_check_done(self, success: bool, message: str, version: str):
        self.check_update_action.setText(tr("menu_check_updates"))
        self.check_update_action.setEnabled(True)

        if not success:
            QMessageBox.warning(self, tr("update_failed"), f"{tr('update_failed_msg')}\n{message}")
            return

        if message == "up_to_date":
            QMessageBox.information(
                self, tr("up_to_date"),
                tr("up_to_date_msg", version=version)
            )
        elif message == "update_available":
            reply = QMessageBox.question(
                self,
                tr("update_available"),
                tr("update_message", version=version, current=APP_VERSION),
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._install_update()
        elif message == "update_installed":
            QMessageBox.information(
                self, tr("update_restart_required"),
                tr("update_restart_msg")
            )

    def _install_update(self):
        """Install the update."""
        self.check_update_action.setEnabled(False)
        self.check_update_action.setText(tr("checking_updates"))

        self._update_worker = UpdateWorker(install=True)
        self._update_worker.finished.connect(self._on_update_check_done)
        self._update_worker.start()

    def _show_about(self):
        QMessageBox.about(
            self,
            tr("about_title"),
            tr("about_text", version=APP_VERSION, repo=GITHUB_REPO)
        )
