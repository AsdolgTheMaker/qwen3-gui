"""
Background worker threads for Qwen3-TTS GUI.
"""

import sys
import io
import re
import torch
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from contextlib import contextmanager

from .constants import MODELS, mode_of
from .settings import get_whisper_model


class _StderrCapture(io.TextIOBase):
    """Capture stderr and parse tqdm progress, forwarding to callback."""

    def __init__(self, callback, original):
        self.callback = callback
        self.original = original
        self._buffer = ""
        self._last_pct = -1

    def write(self, text):
        # Always write to original
        if self.original:
            self.original.write(text)

        # Buffer and look for tqdm patterns
        self._buffer += text

        # tqdm progress pattern: "description:  XX%|" or just "XX%|"
        match = re.search(r'(\d+)%\|', self._buffer)
        if match:
            pct = int(match.group(1))
            # Only report every 5% to avoid flooding
            if pct >= self._last_pct + 5 or pct == 100:
                self._last_pct = pct
                # Try to get filename from the line
                desc_match = re.search(r'([^/\\:\s]+\.(safetensors|bin|model|json|txt)):', self._buffer)
                if desc_match:
                    self.callback(f"Downloading {desc_match.group(1)}: {pct}%")
                else:
                    self.callback(f"Downloading: {pct}%")

        # Clear buffer on carriage return (tqdm uses \r for updates)
        if '\r' in self._buffer or '\n' in self._buffer:
            if '\n' in self._buffer:
                self._last_pct = -1  # Reset for next file
            self._buffer = ""

        return len(text)

    def flush(self):
        if self.original:
            self.original.flush()

    def isatty(self):
        return self.original.isatty() if self.original else False


@contextmanager
def capture_download_progress(callback):
    """Context manager to capture HuggingFace download progress from stderr."""
    original_stderr = sys.stderr
    capture = _StderrCapture(callback, original_stderr)

    try:
        sys.stderr = capture
        yield
    finally:
        sys.stderr = original_stderr


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
        """Request cancellation of the generation."""
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

                with capture_download_progress(self.progress.emit):
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
            else:
                self.progress.emit(f"Using cached {model_label}...")

            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            self.progress.emit("Generating speech...")
            model = self.model_holder["model"]

            # Set seed for reproducibility
            seed = self.params.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

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


class TrainingWorker(QThread):
    """Worker thread for Qwen3-TTS fine-tuning."""

    progress = Signal(str)
    log = Signal(str)
    epoch_progress = Signal(int, int, float)  # epoch, total_epochs, loss
    finished = Signal(bool, str)  # success, message

    # Model name mapping
    MODEL_MAP = {
        0: "Qwen/Qwen3-TTS-12Hz-1.7B-Base",  # 1.7B (recommended)
        1: "Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # 0.6B (faster)
    }

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancelled = False

    def cancel(self):
        """Request cancellation of training."""
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        """Check if training was cancelled."""
        return self._cancelled

    def run(self):
        from pathlib import Path
        import tempfile

        try:
            from .training import prepare_training_data
            from .training.prepare import convert_dataset_to_jsonl
            from .training.trainer import run_training
            from .constants import DATASETS_DIR, MODELS_DIR

            dataset_name = self.params["dataset"]
            model_name = self.params["model_name"]
            base_model_idx = self.params.get("base_model", 0)
            epochs = self.params.get("epochs", 10)
            learning_rate = self.params.get("learning_rate", 2e-6)
            batch_size = self.params.get("batch_size", 2)
            device = self.params.get("device", "cuda:0")

            base_model = self.MODEL_MAP.get(base_model_idx, self.MODEL_MAP[0])
            dataset_dir = DATASETS_DIR / dataset_name
            output_dir = MODELS_DIR / model_name

            self.log.emit(f"Starting training pipeline...")
            self.log.emit(f"Dataset: {dataset_name}")
            self.log.emit(f"Base model: {base_model}")
            self.log.emit(f"Output: {output_dir}")

            # Create temp directory for intermediate files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                raw_jsonl = tmpdir / "train_raw.jsonl"
                prepared_jsonl = tmpdir / "train_prepared.jsonl"

                # Step 1: Convert dataset format
                self.progress.emit("Step 1/3: Converting dataset format...")
                self.log.emit("Converting transcript.txt to JSONL format...")

                try:
                    num_samples = convert_dataset_to_jsonl(
                        dataset_dir=dataset_dir,
                        output_jsonl=raw_jsonl,
                        progress_callback=self.log.emit
                    )
                    self.log.emit(f"Converted {num_samples} samples")
                except Exception as e:
                    self.finished.emit(False, f"Dataset conversion failed: {e}")
                    return

                if self._cancelled:
                    self.finished.emit(False, "Training cancelled")
                    return

                # Step 2: Prepare training data (tokenize audio)
                self.progress.emit("Step 2/3: Preparing training data (tokenizing audio)...")
                self.log.emit("Tokenizing audio files with Qwen3-TTS-Tokenizer...")

                try:
                    prepare_training_data(
                        input_jsonl=raw_jsonl,
                        output_jsonl=prepared_jsonl,
                        device=device,
                        progress_callback=self.log.emit
                    )
                except Exception as e:
                    self.finished.emit(False, f"Data preparation failed: {e}")
                    return

                if self._cancelled:
                    self.finished.emit(False, "Training cancelled")
                    return

                # Step 3: Run training
                self.progress.emit("Step 3/3: Training model...")
                self.log.emit(f"Starting fine-tuning for {epochs} epochs...")

                try:
                    checkpoint = run_training(
                        train_jsonl=prepared_jsonl,
                        output_dir=output_dir,
                        speaker_name=model_name,
                        base_model=base_model,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        num_epochs=epochs,
                        device=device,
                        progress_callback=self.log.emit,
                        epoch_callback=lambda e, t, l: self.epoch_progress.emit(e, t, l),
                        cancel_check=self._is_cancelled
                    )
                except Exception as e:
                    self.finished.emit(False, f"Training failed: {e}")
                    return

                if checkpoint:
                    self.log.emit(f"Training complete!")
                    self.log.emit(f"Model saved to: {checkpoint}")
                    self.finished.emit(True, f"Training complete! Model saved to {checkpoint}")
                else:
                    self.finished.emit(False, "Training was cancelled or failed")

        except ImportError as e:
            self.log.emit(f"Missing dependency: {e}")
            self.log.emit("Please ensure all required packages are installed.")
            self.finished.emit(False, f"Missing dependency: {e}")
        except Exception as e:
            self.log.emit(f"Training error: {e}")
            self.finished.emit(False, str(e))


class TranscriptionWorker(QThread):
    """Worker thread for audio transcription using Whisper."""

    progress = Signal(str)
    result = Signal(int, str)  # row_index, transcription
    finished = Signal(bool, str)  # success, message

    # Shared model holder to avoid reloading
    _model_holder = {"model": None, "processor": None, "model_id": None}

    def __init__(self, audio_files: list, language: str = None):
        """
        Args:
            audio_files: List of (row_index, audio_path) tuples
            language: Optional language hint
        """
        super().__init__()
        self.audio_files = audio_files
        self.language = language
        self._cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def run(self):
        try:
            import librosa

            # Get configured Whisper model
            model_id = get_whisper_model()

            # Lazy load model (reload if model changed)
            if (self._model_holder["model"] is None or
                self._model_holder["processor"] is None or
                self._model_holder["model_id"] != model_id):
                self.progress.emit(f"Loading Whisper ({model_id.split('/')[-1]})...")
                from transformers import WhisperProcessor, WhisperForConditionalGeneration

                with capture_download_progress(self.progress.emit):
                    self._model_holder["processor"] = WhisperProcessor.from_pretrained(model_id)
                    self._model_holder["model"] = WhisperForConditionalGeneration.from_pretrained(model_id)

                self._model_holder["model_id"] = model_id

                if torch.cuda.is_available():
                    self._model_holder["model"] = self._model_holder["model"].to("cuda")

            model = self._model_holder["model"]
            processor = self._model_holder["processor"]

            # Language mapping
            lang_map = {
                "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
                "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt",
                "spanish": "es", "italian": "it", "auto": None
            }
            lang_code = None
            if self.language:
                lang_code = lang_map.get(self.language.lower(), self.language.lower() if len(self.language) <= 3 else None)

            total = len(self.audio_files)
            for i, (row_idx, audio_path) in enumerate(self.audio_files):
                if self._cancelled:
                    self.finished.emit(False, "Cancelled")
                    return

                self.progress.emit(f"Transcribing {i+1}/{total}: {audio_path.split('/')[-1].split(chr(92))[-1]}")

                try:
                    # Load audio
                    audio, sr = librosa.load(audio_path, sr=16000)

                    # Process
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs["input_features"]

                    # Create attention mask (all 1s, same batch size and sequence length)
                    attention_mask = torch.ones(
                        input_features.shape[0],
                        input_features.shape[-1],
                        dtype=torch.long
                    )

                    if torch.cuda.is_available():
                        input_features = input_features.to("cuda")
                        attention_mask = attention_mask.to("cuda")

                    # Generate
                    generate_kwargs = {"attention_mask": attention_mask}
                    if lang_code:
                        generate_kwargs["language"] = lang_code

                    with torch.no_grad():
                        generated_ids = model.generate(input_features, **generate_kwargs)

                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    self.result.emit(row_idx, transcription)

                except Exception as e:
                    self.progress.emit(f"Error transcribing row {row_idx}: {e}")

            self.finished.emit(True, f"Transcribed {total} files")

        except Exception as e:
            self.finished.emit(False, str(e))
