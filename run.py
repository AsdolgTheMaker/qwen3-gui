"""
Qwen3-TTS GUI Launcher
======================
Self-deploying script: creates a venv, installs dependencies,
auto-updates from GitHub, then launches the PySide6-based GUI.

Usage:
    python run.py              # Normal launch with auto-update
    python run.py --no-update  # Skip update check
    python run.py --force-update  # Force re-download even if up to date
"""

import sys
import os
import subprocess
import shutil
import json
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

SCRIPT_DIR = Path(__file__).parent.absolute()

# ---------------------------------------------------------------------------
# CRITICAL: Set HuggingFace cache env vars BEFORE any imports that might
# trigger huggingface_hub or transformers (e.g., qwen_tts).
# These libraries cache the path at import time!
# ---------------------------------------------------------------------------
def _apply_hf_cache_early():
    """Set HF cache env vars from settings before any HF imports."""
    settings_path = SCRIPT_DIR / ".settings.json"
    try:
        if settings_path.exists():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            cache_path = data.get("hf_cache_path", "")
            if cache_path:
                hub_path = str(Path(cache_path) / "hub")
                os.environ["HF_HOME"] = cache_path
                os.environ["HF_HUB_CACHE"] = hub_path
                os.environ["HUGGINGFACE_HUB_CACHE"] = hub_path
                os.environ["TRANSFORMERS_CACHE"] = hub_path
    except Exception:
        pass

_apply_hf_cache_early()
VENV_DIR = SCRIPT_DIR / ".venv"
PACKAGE_DIR = SCRIPT_DIR / "qwen3_gui"
VERSION_FILE = PACKAGE_DIR / "__init__.py"
SETTINGS_FILE = SCRIPT_DIR / ".settings.json"

GITHUB_REPO = "AsdolgTheMaker/qwen3-gui"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"

if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"


# ---------------------------------------------------------------------------
# Settings utilities
# ---------------------------------------------------------------------------

def _get_auto_update_enabled() -> bool:
    """Check if auto-update is enabled in settings."""
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return data.get("auto_update", True)
    except Exception:
        pass
    return True  # Default: enabled


# ---------------------------------------------------------------------------
# Version utilities
# ---------------------------------------------------------------------------

def _get_local_version() -> str:
    """Get the currently installed version."""
    if not VERSION_FILE.exists():
        return "0.0.0"
    try:
        content = VERSION_FILE.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("__version__"):
                # Extract version from: __version__ = "1.0.0"
                return line.split("=")[1].strip().strip('"\'')
    except Exception:
        pass
    return "0.0.0"


def _parse_version(version: str) -> tuple:
    """Parse version string to tuple for comparison."""
    try:
        parts = version.lstrip("v").split(".")
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return (0, 0, 0)


def _version_gt(v1: str, v2: str) -> bool:
    """Check if v1 > v2."""
    return _parse_version(v1) > _parse_version(v2)


# ---------------------------------------------------------------------------
# Auto-updater
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch JSON from URL."""
    # Add cache-busting to avoid GitHub CDN caching
    cache_bust = f"{'&' if '?' in url else '?'}_={int(time.time())}"
    req = Request(url + cache_bust, headers={
        "User-Agent": "Qwen3-TTS-GUI-Updater",
        "Cache-Control": "no-cache",
    })
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, dest: Path, timeout: int = 60):
    """Download file from URL."""
    req = Request(url, headers={"User-Agent": "Qwen3-TTS-GUI-Updater"})
    with urlopen(req, timeout=timeout) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)


def _get_remote_version() -> str:
    """Get the latest version from GitHub releases."""
    try:
        data = _fetch_json(GITHUB_API_URL)
        return data.get("tag_name", "").lstrip("v")
    except Exception:
        # If no releases, try to get version from main branch via GitHub Contents API
        # (raw.githubusercontent.com has aggressive CDN caching)
        try:
            import base64
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/qwen3_gui/__init__.py"
            data = _fetch_json(api_url)
            content = base64.b64decode(data["content"]).decode("utf-8")
            for line in content.splitlines():
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
        except Exception:
            pass
    return "0.0.0"


def _perform_update() -> bool:
    """Download and install the latest version. Returns True if successful."""
    print("[update] Downloading latest version...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "repo.zip"

        try:
            _download_file(GITHUB_ZIP_URL, zip_path)
        except Exception as e:
            print(f"[update] Download failed: {e}")
            return False

        print("[update] Extracting...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
        except Exception as e:
            print(f"[update] Extraction failed: {e}")
            return False

        # Find the extracted folder (usually repo-name-main)
        extracted_dirs = [d for d in tmpdir.iterdir() if d.is_dir()]
        if not extracted_dirs:
            print("[update] No directory found in archive")
            return False

        extracted_root = extracted_dirs[0]
        new_package = extracted_root / "qwen3_gui"

        if not new_package.exists():
            print("[update] Package folder not found in archive")
            return False

        # Backup current package (just in case)
        backup_dir = SCRIPT_DIR / ".qwen3_gui_backup"
        if PACKAGE_DIR.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(PACKAGE_DIR), str(backup_dir))

        # Install new package
        try:
            shutil.copytree(str(new_package), str(PACKAGE_DIR))
            # Remove backup on success
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            print("[update] Update successful!")
            return True
        except Exception as e:
            print(f"[update] Installation failed: {e}")
            # Restore backup
            if backup_dir.exists():
                if PACKAGE_DIR.exists():
                    shutil.rmtree(PACKAGE_DIR)
                shutil.move(str(backup_dir), str(PACKAGE_DIR))
            return False


def check_for_updates(force: bool = False) -> bool:
    """
    Check for updates and install if available.
    Returns True if an update was installed.
    """
    local_version = _get_local_version()
    print(f"[update] Current version: {local_version}")

    try:
        print("[update] Checking for updates...")
        remote_version = _get_remote_version()

        if remote_version == "0.0.0":
            print("[update] Could not determine remote version, skipping update")
            return False

        print(f"[update] Latest version: {remote_version}")

        if force or _version_gt(remote_version, local_version):
            if force:
                print("[update] Forcing update...")
            else:
                print(f"[update] New version available: {remote_version}")
            return _perform_update()
        else:
            print("[update] Already up to date")
            return False

    except URLError as e:
        print(f"[update] Network error (offline?): {e.reason}")
        return False
    except Exception as e:
        print(f"[update] Update check failed: {e}")
        return False


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
    # Check if all critical dependencies are installed
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import PySide6; import qwen_tts; import sounddevice"],
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
        "qwen-tts",
        "PySide6",
        "PySide6-Addons",  # Includes QtMultimedia plugins
        "soundfile",
        "sounddevice",  # For reliable audio playback
        "librosa",
        "requests",
    ])

    # Try to install flash-attn for faster inference (optional, CUDA only)
    if cuda:
        print("[setup] Attempting to install flash-attn (optional, may fail) ...")
        result = subprocess.run(
            [str(VENV_PIP), "install", "flash-attn", "--no-build-isolation"],
            capture_output=True,
        )
        if result.returncode == 0:
            print("[setup] flash-attn installed successfully")
        else:
            print("[setup] flash-attn installation failed (this is optional, continuing...)")

    print("[setup] Dependencies installed successfully.")


def bootstrap(skip_update: bool = False, force_update: bool = False):
    """Bootstrap the application environment."""
    if _in_venv():
        # Already in venv, just check deps
        try:
            import PySide6  # noqa: F401
            import qwen_tts  # noqa: F401
            import sounddevice  # noqa: F401
        except ImportError:
            print("[setup] Missing dependencies, installing ...")
            subprocess.check_call([
                str(VENV_PIP), "install",
                "qwen-tts", "PySide6", "soundfile", "sounddevice", "librosa", "requests"
            ])
        return

    # Check for updates before setting up venv (uses system Python's urllib)
    if not skip_update and _get_auto_update_enabled():
        check_for_updates(force=force_update)
    elif not skip_update and not _get_auto_update_enabled():
        print("[update] Auto-update disabled in settings, skipping")

    # Not in venv - set up and re-exec
    _ensure_venv()
    _ensure_deps()

    # Pass through the update flags
    args = [str(VENV_PYTHON), str(Path(__file__).absolute())]
    args.append("--no-update")  # Already updated, don't check again
    args.extend([a for a in sys.argv[1:] if a not in ("--no-update", "--force-update")])

    print("[setup] Launching inside virtual environment ...")
    os.execv(str(VENV_PYTHON), args)


# ---------------------------------------------------------------------------
# Bootstrap and launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse arguments
    skip_update = "--no-update" in sys.argv
    force_update = "--force-update" in sys.argv

    bootstrap(skip_update=skip_update, force_update=force_update)

    # Import and run the application
    from qwen3_gui import main
    main()
