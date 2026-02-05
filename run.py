"""
Qwen3-TTS GUI Launcher
======================
Self-deploying script: creates a venv, installs dependencies,
then launches the PySide6-based GUI for text-to-speech generation.

Usage:  python run.py
"""

import sys
import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_DIR = SCRIPT_DIR / ".venv"

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
    """Check if we're running inside our project venv."""
    return sys.prefix != sys.base_prefix and VENV_DIR.is_dir()


def _has_cuda() -> bool:
    """Check if NVIDIA GPU is available."""
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
    """Create virtual environment if it doesn't exist."""
    if VENV_PYTHON.is_file():
        return
    print("[setup] Creating virtual environment ...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])


def _ensure_deps():
    """Install dependencies if not already installed."""
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
    subprocess.check_call([
        str(VENV_PIP), "install",
        "qwen-tts", "PySide6", "soundfile", "librosa", "requests"
    ])

    print("[setup] Dependencies installed successfully.")


def bootstrap():
    """Bootstrap the application environment."""
    if _in_venv():
        # Already in venv, just check deps
        try:
            import PySide6  # noqa: F401
            import qwen_tts  # noqa: F401
        except ImportError:
            print("[setup] Missing dependencies, installing ...")
            subprocess.check_call([
                str(VENV_PIP), "install",
                "qwen-tts", "PySide6", "soundfile", "librosa", "requests"
            ])
        return

    # Not in venv - set up and re-exec
    _ensure_venv()
    _ensure_deps()

    print("[setup] Launching inside virtual environment ...")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).absolute())] + sys.argv[1:])


# ---------------------------------------------------------------------------
# Bootstrap and launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bootstrap()

    # Import and run the application
    from qwen3_gui import main
    main()
