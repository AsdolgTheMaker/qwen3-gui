"""
Output log widget for displaying application messages and progress.
"""

from datetime import datetime

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QTextCursor, QColor

from ..translations import tr


class OutputLogWidget(QFrame):
    """Output log panel for displaying status messages and generation progress."""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header with title and clear button
        header_layout = QHBoxLayout()

        title = QLabel(tr("output_log"))
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        clear_btn = QPushButton(tr("clear"))
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(self.clear)
        header_layout.addWidget(clear_btn)

        layout.addLayout(header_layout)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.log_text)

        # Initial message
        self.log_info(tr("log_initialized"))
        self.log_info(tr("log_ready"))

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _append_html(self, html: str):
        """Append HTML to the log and scroll to bottom."""
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertHtml(html + "<br>")
        self.log_text.moveCursor(QTextCursor.End)

    @Slot()
    def clear(self):
        """Clear the log."""
        self.log_text.clear()

    @Slot(str)
    def log(self, message: str):
        """Log a plain message."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #d4d4d4;">{message}</span>'
        )

    @Slot(str)
    def log_info(self, message: str):
        """Log an info message (blue)."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #569cd6;">[INFO]</span> '
            f'<span style="color: #d4d4d4;">{message}</span>'
        )

    @Slot(str)
    def log_success(self, message: str):
        """Log a success message (green)."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #4ec9b0;">[SUCCESS]</span> '
            f'<span style="color: #4ec9b0;">{message}</span>'
        )

    @Slot(str)
    def log_warning(self, message: str):
        """Log a warning message (yellow)."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #dcdcaa;">[WARNING]</span> '
            f'<span style="color: #dcdcaa;">{message}</span>'
        )

    @Slot(str)
    def log_error(self, message: str):
        """Log an error message (red)."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #f14c4c;">[ERROR]</span> '
            f'<span style="color: #f14c4c;">{message}</span>'
        )

    @Slot(str)
    def log_progress(self, message: str):
        """Log a progress message (purple)."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #c586c0;">[PROGRESS]</span> '
            f'<span style="color: #d4d4d4;">{message}</span>'
        )

    @Slot(str, str)
    def log_model(self, model_name: str, action: str):
        """Log model-related activity."""
        timestamp = self._timestamp()
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #4fc1ff;">[MODEL]</span> '
            f'<span style="color: #ce9178;">{model_name}</span>: '
            f'<span style="color: #d4d4d4;">{action}</span>'
        )

    @Slot(str, int)
    def log_audio_saved(self, path: str, sample_rate: int = 0):
        """Log audio file saved."""
        timestamp = self._timestamp()
        sr_text = f" ({sample_rate} Hz)" if sample_rate else ""
        self._append_html(
            f'<span style="color: #6a9955;">[{timestamp}]</span> '
            f'<span style="color: #4ec9b0;">[SAVED]</span> '
            f'<span style="color: #d4d4d4;">Audio saved to </span>'
            f'<span style="color: #ce9178;">{path}</span>'
            f'<span style="color: #6a9955;">{sr_text}</span>'
        )
