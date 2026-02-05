"""
Dataset class for Qwen3-TTS fine-tuning.

Based on the official Qwen3-TTS finetuning implementation.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Union
from pathlib import Path


def load_audio(audio_input: Union[str, Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
    """Load audio from file path or return if already loaded."""
    if isinstance(audio_input, str):
        import librosa
        audio, sr = librosa.load(audio_input, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)
    elif isinstance(audio_input, (tuple, list)):
        return audio_input[0].astype(np.float32), int(audio_input[1])
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")


class TTSDataset(Dataset):
    """Dataset for TTS fine-tuning with pre-tokenized audio codes."""

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        processor: Any,
        config: Any,
    ):
        """
        Args:
            data_list: List of dicts with keys: text, audio_codes, ref_audio, language (optional)
            processor: Qwen3TTS processor for text tokenization
            config: Qwen3TTS config object
        """
        self.data_list = data_list
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.data_list)

    def _build_assistant_text(self, text: str) -> str:
        """Build the assistant text format expected by the model."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using the processor."""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids

    @torch.inference_mode()
    def _extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Extract mel spectrogram from audio using the official method."""
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        import librosa

        # Resample to 24kHz if needed
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_list[idx]

        # Get text and build assistant format
        text = item["text"]
        text = self._build_assistant_text(text)
        text_ids = self._tokenize_text(text)

        # Get audio codes (pre-tokenized)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        # Load reference audio and extract mel spectrogram
        ref_audio_path = item["ref_audio"]
        wav, sr = load_audio(ref_audio_path)
        ref_mel = self._extract_mels(wav, sr)

        return {
            "text_ids": text_ids[:, :-5],  # Remove last 5 tokens as per official impl
            "audio_codes": audio_codes,    # Shape: [time, 16]
            "ref_mel": ref_mel
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching samples - matches official implementation."""
        # Calculate sequence lengths
        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        # Initialize tensors
        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:, 0]
            audio_codecs = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # Text channel (channel 0)
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0, 3:]
            input_ids[i, 8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8+text_ids_len+codec_ids_len] = True

            # Codec channel (channel 1)
            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,  # placeholder for speaker embedding
                self.config.talker_config.codec_pad_id
            ])
            input_ids[i, 8:8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            # Labels for codec 0 prediction
            codec_0_labels[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            # All codec channels
            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, :] = audio_codecs

            # Masks
            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # speaker embedding position
            codec_mask[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True

        # Stack reference mels
        ref_mels = torch.cat([data['ref_mel'] for data in batch], dim=0)

        return {
            'input_ids': input_ids,
            'codec_ids': codec_ids,
            'ref_mels': ref_mels,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'attention_mask': attention_mask,
            'codec_0_labels': codec_0_labels,
            'codec_mask': codec_mask
        }


def load_jsonl_data(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
