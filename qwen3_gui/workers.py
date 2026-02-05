"""
Background worker threads for Qwen3-TTS GUI.
"""

import torch
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from .constants import MODELS, mode_of


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
