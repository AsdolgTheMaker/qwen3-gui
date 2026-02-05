"""
Qwen3-TTS GUI
=============
Modern PySide6-based interface for Qwen3 Text-to-Speech.

Features:
  - Text-to-Speech with Custom Voice, Voice Design, and Voice Clone modes
  - Voice model training from custom datasets
  - Dataset builder tools
  - Built-in media player for instant playback
  - Auto-update support

Usage:  python run.py
"""

import sys
import os
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_DIR = SCRIPT_DIR / ".venv"
CONFIG_FILE = SCRIPT_DIR / "config.json"
DATASETS_DIR = SCRIPT_DIR / "datasets"
MODELS_DIR = SCRIPT_DIR / "trained_models"
OUTPUT_DIR = SCRIPT_DIR / "output"

if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"


# ---------------------------------------------------------------------------
# Bootstrap: ensure we are running inside the project venv with deps installed
# ---------------------------------------------------------------------------

def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix and VENV_DIR.is_dir()


def _has_cuda() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _gpu_needs_cu128() -> bool:
    """Check if GPU requires CUDA 12.8+ (Blackwell / sm_120+)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.strip().splitlines():
            major, _, minor = line.strip().partition(".")
            if int(major) >= 12:
                return True
    except Exception:
        pass
    return False


def _ensure_venv():
    if VENV_PYTHON.is_file():
        return
    print("[setup] Creating virtual environment ...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])


def _ensure_deps():
    # Check if PySide6 is installed (our main GUI dependency)
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import PySide6; import qwen_tts"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return

    print("[setup] Installing dependencies (this may take a while on first run) ...")

    cuda = _has_cuda()
    if cuda:
        if _gpu_needs_cu128():
            cuda_tag = "cu128"
            print("[setup] Blackwell GPU detected - installing PyTorch with CUDA 12.8 ...")
        else:
            cuda_tag = "cu124"
            print("[setup] NVIDIA GPU detected - installing PyTorch with CUDA 12.4 ...")
        subprocess.check_call([
            str(VENV_PIP), "install",
            "torch", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
        ])
    else:
        print("[setup] No NVIDIA GPU detected - installing CPU-only PyTorch ...")
        subprocess.check_call([str(VENV_PIP), "install", "torch", "torchaudio"])

    print("[setup] Installing qwen-tts and PySide6 ...")
    subprocess.check_call([str(VENV_PIP), "install", "qwen-tts", "PySide6", "soundfile", "librosa", "requests"])

    print("[setup] Dependencies installed successfully.")


def bootstrap():
    if _in_venv():
        try:
            import PySide6  # noqa: F401
            import qwen_tts  # noqa: F401
        except ImportError:
            print("[setup] Missing dependencies, installing ...")
            subprocess.check_call([str(VENV_PIP), "install", "qwen-tts", "PySide6", "soundfile", "librosa", "requests"])
        return

    _ensure_venv()
    _ensure_deps()

    print("[setup] Launching inside virtual environment ...")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).absolute())] + sys.argv[1:])


# ---------------------------------------------------------------------------
bootstrap()
# ---------------------------------------------------------------------------

import threading
import wave
from typing import Optional, Callable

import torch
import soundfile as sf
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QLineEdit, QSlider, QCheckBox,
    QProgressBar, QTabWidget, QGroupBox, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QScrollArea, QToolTip, QSizePolicy, QStyle, QStyleFactory
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QUrl, QTimer, QSize, QSettings
)
from PySide6.QtGui import (
    QFont, QIcon, QPalette, QColor, QAction, QKeySequence
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


# ---------------------------------------------------------------------------
# Application version and update info
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"
GITHUB_REPO = "AsdolgTheMaker/qwen3-gui"
UPDATE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


# ---------------------------------------------------------------------------
# ELI5 Tooltips - Beginner-friendly explanations
# ---------------------------------------------------------------------------
TOOLTIPS = {
    # Model selection
    "model": """<b>Which AI brain to use</b><br><br>
Think of these like different voice actors with different skills:<br><br>
<b>Custom Voice</b> - Uses built-in voice presets (like choosing a character)<br>
<b>Voice Design</b> - You describe what voice you want in plain English<br>
<b>Voice Clone</b> - Copies someone's voice from an audio sample<br><br>
<i>1.7B models sound better but need more computer power.<br>
0.6B models are faster but slightly lower quality.</i>""",

    # Language
    "language": """<b>What language should it speak?</b><br><br>
'Auto' means the AI figures it out automatically from your text.<br><br>
Setting it manually can sometimes sound more natural,<br>
especially for languages that share similar characters.""",

    # Speaker
    "speaker": """<b>Pick a voice character</b><br><br>
Each speaker has a unique voice personality.<br>
Some are better suited for certain languages<br>
(shown in the description), but all can speak<br>
any supported language.""",

    # Text prompt
    "text_prompt": """<b>What should the voice say?</b><br><br>
Type or paste any text here and the AI will<br>
read it out loud in the selected voice.<br><br>
<i>Tip: Punctuation affects how it sounds!<br>
Periods = pauses, ! = emphasis, ? = rising tone</i>""",

    # Instruction
    "instruction": """<b>How should they say it?</b><br><br>
Give directions like you're coaching an actor:<br>
- "Speak happily and excited"<br>
- "Sound tired and sleepy"<br>
- "Read this like a news anchor"<br><br>
<i>For Voice Design mode, describe the voice itself:<br>
"Young woman with a warm, friendly tone"</i>""",

    # Reference audio
    "ref_audio": """<b>Voice sample to copy</b><br><br>
Upload a recording of the voice you want to clone.<br><br>
<i>Best results with:</i><br>
- 3-10 seconds of clear speech<br>
- No background music or noise<br>
- Single speaker only""",

    # Reference text
    "ref_text": """<b>What's being said in the sample?</b><br><br>
Type exactly what the person says in your<br>
reference audio. This helps the AI understand<br>
how that voice pronounces things.<br><br>
<i>Can skip this if you enable 'x-vector only' mode,<br>
but quality may be lower.</i>""",

    # X-vector
    "xvector": """<b>Simple voice copying mode</b><br><br>
When ON: Just copies the general "sound" of the voice<br>
(like the pitch and tone) without needing the transcript.<br><br>
When OFF: Does deeper analysis using the transcript<br>
for more accurate cloning.<br><br>
<i>Try ON if you don't know what's said in the sample.</i>""",

    # Temperature
    "temperature": """<b>Creativity vs Consistency</b><br><br>
Like a creativity dial for the AI:<br><br>
<b>Low (0.1-0.5):</b> Very consistent, predictable<br>
<b>Medium (0.6-1.0):</b> Natural variation (recommended)<br>
<b>High (1.1+):</b> More expressive but might sound weird<br><br>
<i>Default: 0.9 - a good balance</i>""",

    # Top-K
    "top_k": """<b>Word choice variety</b><br><br>
Limits how many options the AI considers when<br>
deciding what sound comes next.<br><br>
<b>Lower numbers:</b> Safer, more predictable<br>
<b>Higher numbers:</b> More varied, might surprise you<br><br>
<i>Default: 50 - works well for most cases</i>""",

    # Top-P
    "top_p": """<b>Smart word filtering</b><br><br>
Only keeps the most likely sounds until their<br>
combined probability reaches this threshold.<br><br>
<b>1.0:</b> Use all options (disabled)<br>
<b>0.9:</b> Only top 90% most likely<br><br>
<i>Default: 1.0 - usually best to leave this alone</i>""",

    # Repetition penalty
    "rep_penalty": """<b>Avoid repetitive sounds</b><br><br>
Prevents the AI from getting "stuck" repeating<br>
the same sounds over and over.<br><br>
<b>1.0:</b> No penalty (might loop)<br>
<b>1.05:</b> Light penalty (recommended)<br>
<b>1.5+:</b> Strong penalty (might sound choppy)<br><br>
<i>Default: 1.05</i>""",

    # Max tokens
    "max_tokens": """<b>Maximum audio length</b><br><br>
Limits how long the generated audio can be.<br>
Higher = longer possible output but uses more memory.<br><br>
<i>2048 tokens ~ 2-3 minutes of speech<br>
4096 tokens ~ 5-6 minutes of speech</i>""",

    # Dtype
    "dtype": """<b>Number precision (advanced)</b><br><br>
How precisely the AI does math internally:<br><br>
<b>bfloat16:</b> Fast & efficient (needs newer GPU)<br>
<b>float16:</b> Good for older GPUs<br>
<b>float32:</b> Most compatible but slower<br><br>
<i>If you get errors, try float16 or float32</i>""",

    # Flash attention
    "flash_attn": """<b>Speed boost (advanced)</b><br><br>
A faster way to process the AI model.<br>
Needs a newer NVIDIA GPU (RTX 30xx or newer).<br><br>
<i>If it causes errors, the app will automatically<br>
fall back to the normal (slower) method.</i>""",

    # Dataset - audio folder
    "dataset_audio": """<b>Folder with voice recordings</b><br><br>
Point this to a folder containing audio files<br>
(.wav, .mp3, .flac) of the voice you want to train.<br><br>
<i>Best results with:</i><br>
- 30+ minutes of audio total<br>
- Clear speech, minimal background noise<br>
- Consistent recording quality""",

    # Dataset - transcript
    "dataset_transcript": """<b>Text file with what's said</b><br><br>
A text file matching audio to transcripts.<br><br>
Format: <code>filename|transcript text</code><br>
Example: <code>audio001.wav|Hello, how are you?</code><br><br>
<i>One line per audio file.</i>""",

    # Training epochs
    "epochs": """<b>Training repetitions</b><br><br>
How many times the AI studies your dataset.<br><br>
<b>More epochs:</b> Better learning, takes longer<br>
<b>Fewer epochs:</b> Faster, might not learn well<br><br>
<i>Start with 10-20, increase if results aren't good.</i>""",

    # Learning rate
    "learning_rate": """<b>Learning speed</b><br><br>
How big of adjustments the AI makes while learning.<br><br>
<b>Too high:</b> Learns fast but might "overshoot"<br>
<b>Too low:</b> Very slow, might get stuck<br><br>
<i>Default: 0.0001 - a safe starting point</i>""",

    # Batch size
    "batch_size": """<b>Samples per lesson</b><br><br>
How many audio clips the AI looks at simultaneously.<br><br>
<b>Larger:</b> Faster training, needs more GPU memory<br>
<b>Smaller:</b> Slower but works on modest hardware<br><br>
<i>Start with 4, increase if you have lots of VRAM</i>""",
}


def set_tooltip(widget: QWidget, key: str):
    """Set an ELI5 tooltip on a widget."""
    if key in TOOLTIPS:
        widget.setToolTip(TOOLTIPS[key])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = {
    "Custom Voice (1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Custom Voice (0.6B)": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Voice Design (1.7B)": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Voice Clone (1.7B)":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Voice Clone (0.6B)":  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

SPEAKER_INFO = {
    "Vivian":   "Bright, slightly edgy young female (Chinese)",
    "Serena":   "Warm, gentle young female (Chinese)",
    "Uncle_Fu": "Seasoned male, low mellow timbre (Chinese)",
    "Dylan":    "Youthful Beijing male, clear natural (Chinese - Beijing)",
    "Eric":     "Lively Chengdu male, slightly husky (Chinese - Sichuan)",
    "Ryan":     "Dynamic male, strong rhythmic drive (English)",
    "Aiden":    "Sunny American male, clear midrange (English)",
    "Ono_Anna": "Playful female, light nimble timbre (Japanese)",
    "Sohee":    "Warm female, rich emotion (Korean)",
}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

DTYPE_OPTIONS = ["bfloat16", "float16", "float32"]


def mode_of(label: str) -> str:
    """Determine mode from model label."""
    if "Custom Voice" in label:
        return "custom"
    if "Voice Design" in label:
        return "design"
    return "clone"


# ---------------------------------------------------------------------------
# Worker threads for background tasks
# ---------------------------------------------------------------------------

class GenerationWorker(QThread):
    """Worker thread for TTS generation."""
    progress = Signal(str)
    finished = Signal(bool, str, str)  # success, message, output_path

    def __init__(self, params: dict, model_holder: dict):
        super().__init__()
        self.params = params
        self.model_holder = model_holder
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from qwen_tts import Qwen3TTSModel

            model_label = self.params["model_label"]
            model_id = MODELS[model_label]
            mode = mode_of(model_label)

            # Load or reuse model
            if self.model_holder.get("model") is None or self.model_holder.get("model_id") != model_id:
                self.progress.emit(f"Loading {model_label}...")

                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                dtype = dtype_map.get(self.params["dtype"], torch.bfloat16)

                kwargs = {
                    "device_map": self.params["device"],
                    "dtype": dtype,
                }

                if self.params.get("flash_attn"):
                    kwargs["attn_implementation"] = "flash_attention_2"

                try:
                    model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
                except Exception:
                    if "attn_implementation" in kwargs:
                        self.progress.emit("Flash Attention unavailable, retrying...")
                        del kwargs["attn_implementation"]
                        model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
                    else:
                        raise

                self.model_holder["model"] = model
                self.model_holder["model_id"] = model_id

            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            self.progress.emit("Generating speech...")
            model = self.model_holder["model"]

            gen_kwargs = {
                "max_new_tokens": self.params["max_tokens"],
                "do_sample": True,
                "top_k": self.params["top_k"],
                "top_p": self.params["top_p"],
                "temperature": self.params["temperature"],
                "repetition_penalty": self.params["rep_penalty"],
            }

            text = self.params["text"]
            lang = self.params["language"]

            if mode == "custom":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=self.params["speaker"],
                    instruct=self.params.get("instruction") or None,
                    **gen_kwargs,
                )
            elif mode == "design":
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=self.params["instruction"],
                    **gen_kwargs,
                )
            else:  # clone
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=self.params["ref_audio"],
                    ref_text=self.params.get("ref_text") or None,
                    x_vector_only_mode=self.params.get("xvector", False),
                    **gen_kwargs,
                )

            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            self.progress.emit("Saving audio...")
            out_path = self.params["output_path"]
            sf.write(out_path, wavs[0], sr)

            self.finished.emit(True, f"Saved to {out_path}", out_path)

        except Exception as e:
            self.finished.emit(False, str(e), "")


# ---------------------------------------------------------------------------
# Media Player Widget
# ---------------------------------------------------------------------------

class MediaPlayerWidget(QFrame):
    """Built-in media player for testing generated audio."""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._setup_ui()
        self._current_file = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Audio Player")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(title)

        # File label
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #666;")
        layout.addWidget(self.file_label)

        # Player setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)

        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self._seek)
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

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        btn_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()

        # Volume
        btn_layout.addWidget(QLabel("Vol:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self._set_volume)
        btn_layout.addWidget(self.volume_slider)

        layout.addLayout(btn_layout)

        # Connect signals
        self.player.positionChanged.connect(self._update_position)
        self.player.durationChanged.connect(self._update_duration)
        self.player.playbackStateChanged.connect(self._state_changed)

    def load_file(self, path: str):
        """Load an audio file for playback."""
        self._current_file = path
        self.file_label.setText(Path(path).name)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress_slider.setEnabled(True)

    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop(self):
        self.player.stop()

    def _seek(self, position: int):
        self.player.setPosition(position)

    def _set_volume(self, value: int):
        self.audio_output.setVolume(value / 100.0)

    def _update_position(self, position: int):
        self.progress_slider.setValue(position)
        mins, secs = divmod(position // 1000, 60)
        self.time_current.setText(f"{mins}:{secs:02d}")

    def _update_duration(self, duration: int):
        self.progress_slider.setRange(0, duration)
        mins, secs = divmod(duration // 1000, 60)
        self.time_total.setText(f"{mins}:{secs:02d}")

    def _state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setText("Pause")
        else:
            self.btn_play.setText("Play")


# ---------------------------------------------------------------------------
# TTS Tab
# ---------------------------------------------------------------------------

class TTSTab(QWidget):
    """Main text-to-speech interface."""

    def __init__(self, media_player: MediaPlayerWidget):
        super().__init__()
        self.media_player = media_player
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

    def _on_generate(self):
        error = self._validate()
        if error:
            QMessageBox.warning(self, "Validation Error", error)
            return

        # Ensure output directory exists
        out_path = Path(self.output_path_edit.text())
        out_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "model_label": self.model_combo.currentText(),
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

    def _on_progress(self, message: str):
        self.status_label.setText(message)

    def _on_finished(self, success: bool, message: str, output_path: str):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.hide()
        self.status_label.setText(message)

        if success and output_path:
            self.media_player.load_file(output_path)


# ---------------------------------------------------------------------------
# Dataset Builder Tab
# ---------------------------------------------------------------------------

class DatasetBuilderTab(QWidget):
    """Interface for building voice training datasets."""

    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.dataset_entries = []

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


# ---------------------------------------------------------------------------
# Training Tab
# ---------------------------------------------------------------------------

class TrainingTab(QWidget):
    """Interface for training custom voice models."""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info
        info_label = QLabel(
            "<b>Voice Model Training</b><br>"
            "Train a custom voice model using your dataset.<br>"
            "<i>Note: Training requires significant GPU memory and time.</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Dataset selection
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QGridLayout(dataset_group)

        dataset_layout.addWidget(QLabel("Select Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self._refresh_datasets()
        dataset_layout.addWidget(self.dataset_combo, 0, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_datasets)
        dataset_layout.addWidget(refresh_btn, 0, 2)

        layout.addWidget(dataset_group)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)

        row = 0

        # Base model
        params_layout.addWidget(QLabel("Base Model:"), row, 0)
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems([
            "Qwen3-TTS-1.7B (recommended)",
            "Qwen3-TTS-0.6B (faster)"
        ])
        params_layout.addWidget(self.base_model_combo, row, 1)
        row += 1

        # Epochs
        epochs_label = QLabel("Epochs:")
        set_tooltip(epochs_label, "epochs")
        params_layout.addWidget(epochs_label, row, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin, row, 1)
        row += 1

        # Learning rate
        lr_label = QLabel("Learning Rate:")
        set_tooltip(lr_label, "learning_rate")
        params_layout.addWidget(lr_label, row, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.000001, 0.01)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.00001)
        self.lr_spin.setValue(0.0001)
        params_layout.addWidget(self.lr_spin, row, 1)
        row += 1

        # Batch size
        batch_label = QLabel("Batch Size:")
        set_tooltip(batch_label, "batch_size")
        params_layout.addWidget(batch_label, row, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        params_layout.addWidget(self.batch_spin, row, 1)
        row += 1

        # Output model name
        params_layout.addWidget(QLabel("Output Model Name:"), row, 0)
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("my_custom_voice")
        params_layout.addWidget(self.model_name_edit, row, 1)

        layout.addWidget(params_group)

        # Training controls
        controls_layout = QHBoxLayout()

        self.train_btn = QPushButton("Start Training")
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

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Training log
        log_group = QGroupBox("Training Log")
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

    def _start_training(self):
        dataset = self.dataset_combo.currentText()
        if not dataset:
            QMessageBox.warning(self, "No Dataset", "Please select or create a dataset first.")
            return

        model_name = self.model_name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, "No Model Name", "Please enter a name for the output model.")
            return

        self.log_text.clear()
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
        self.log_text.append(f"Dataset: {dataset}")
        self.log_text.append(f"Epochs: {self.epochs_spin.value()}")
        self.log_text.append(f"Learning Rate: {self.lr_spin.value()}")
        self.log_text.append(f"Batch Size: {self.batch_spin.value()}")
        self.log_text.append("")
        self.log_text.append("NOTE: Full training implementation requires additional setup.")
        self.log_text.append("This feature is a placeholder for the training pipeline.")
        self.log_text.append("See qwen-tts documentation for training instructions.")

        # TODO: Implement actual training pipeline
        # This would typically involve:
        # 1. Loading the dataset
        # 2. Setting up the training loop
        # 3. Fine-tuning the model
        # 4. Saving checkpoints


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

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

        # Media player (shared)
        self.media_player = MediaPlayerWidget()

        # TTS tab
        self.tts_tab = TTSTab(self.media_player)
        self.tabs.addTab(self.tts_tab, "Text-to-Speech")

        # Dataset builder tab
        self.dataset_tab = DatasetBuilderTab()
        self.tabs.addTab(self.dataset_tab, "Dataset Builder")

        # Training tab
        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "Voice Training")

        splitter.addWidget(self.tabs)

        # Right side - media player
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.media_player)
        right_layout.addStretch()

        splitter.addWidget(right_panel)

        # Set splitter sizes (70% tabs, 30% player)
        splitter.setSizes([700, 300])

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
                        import webbrowser
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


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main():
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Enable High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    # Set application info
    app.setApplicationName("Qwen3-TTS GUI")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("AsdolgTheMaker")

    # Enable rich tooltips
    QToolTip.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
