"""
Background worker threads for Qwen3-TTS GUI.
"""

import torch
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from contextlib import contextmanager

from .constants import MODELS, mode_of
from .settings import get_whisper_model


def _format_size(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


@contextmanager
def redirect_hf_progress(callback):
    """
    Context manager to redirect HuggingFace download progress to a callback.

    Args:
        callback: Function that takes a message string
    """
    try:
        from huggingface_hub import utils as hf_utils

        # Store original tqdm class
        original_tqdm = hf_utils.tqdm.tqdm

        # Create a custom tqdm that reports to our callback
        class CallbackTqdm(original_tqdm):
            def __init__(self, *args, **kwargs):
                # Extract description for our messages
                self._desc = kwargs.get('desc', '')
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                if self.total:
                    pct = 100 * self.n / self.total
                    size_info = f"{_format_size(self.n)}/{_format_size(self.total)}"
                    msg = f"Downloading {self._desc}: {pct:.0f}% ({size_info})"
                    callback(msg)

            def close(self):
                if self.total and self.n >= self.total:
                    callback(f"Downloaded {self._desc}: {_format_size(self.total)}")
                super().close()

        # Monkey-patch
        hf_utils.tqdm.tqdm = CallbackTqdm

        yield

    except ImportError:
        # huggingface_hub not available, just yield
        yield
    finally:
        # Restore original if we patched it
        try:
            hf_utils.tqdm.tqdm = original_tqdm
        except (NameError, UnboundLocalError):
            pass


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

                # Redirect download progress to our signal
                with redirect_hf_progress(self.progress.emit):
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
    """Worker thread for model training (placeholder)."""

    progress = Signal(str)
    log = Signal(str)
    finished = Signal(bool, str)  # success, message

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancelled = False

    def cancel(self):
        """Request cancellation of training."""
        self._cancelled = True

    def run(self):
        # TODO: Implement actual training pipeline
        # This would typically involve:
        # 1. Loading the dataset
        # 2. Setting up the training loop with the base model
        # 3. Fine-tuning with the specified parameters
        # 4. Saving checkpoints

        self.log.emit("Training implementation is a placeholder.")
        self.log.emit("See qwen-tts documentation for training instructions.")
        self.finished.emit(False, "Training not yet implemented")


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
            if self._model_holder["model"] is None or self._model_holder["model_id"] != model_id:
                self.progress.emit(f"Loading Whisper model ({model_id.split('/')[-1]})...")
                from transformers import WhisperProcessor, WhisperForConditionalGeneration

                # Redirect download progress to our signal
                with redirect_hf_progress(self.progress.emit):
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
