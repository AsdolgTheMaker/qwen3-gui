"""
Dataset class for Qwen3-TTS fine-tuning.

Based on the official Qwen3-TTS finetuning implementation.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path


def load_audio(audio_input: Union[str, Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
    """Load audio from file path or return if already loaded."""
    if isinstance(audio_input, str):
        import soundfile as sf
        waveform, sr = sf.read(audio_input)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform.astype(np.float32), sr
    elif isinstance(audio_input, (tuple, list)):
        return audio_input[0].astype(np.float32), audio_input[1]
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    target_sr: int = 24000
) -> np.ndarray:
    """Extract mel spectrogram from audio waveform."""
    import librosa

    # Resample if necessary
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db.astype(np.float32)


class TTSDataset(Dataset):
    """Dataset for TTS fine-tuning with pre-tokenized audio codes."""

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        processor: Any,
        max_text_length: int = 512,
        max_codec_length: int = 2048
    ):
        """
        Args:
            data_list: List of dicts with keys: text, audio_codes, ref_audio, language (optional)
            processor: Qwen3TTS processor for text tokenization
            max_text_length: Maximum text token length
            max_codec_length: Maximum audio codec length
        """
        self.data_list = data_list
        self.processor = processor
        self.max_text_length = max_text_length
        self.max_codec_length = max_codec_length

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_list[idx]

        # Get text and tokenize
        text = item["text"]
        language = item.get("language", "auto")

        # Tokenize text
        text_tokens = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_text_length
        )

        # Get audio codes (pre-tokenized)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        # Truncate if too long
        if audio_codes.shape[0] > self.max_codec_length:
            audio_codes = audio_codes[:self.max_codec_length]

        # Load reference audio and extract mel spectrogram
        ref_audio = item["ref_audio"]
        waveform, sr = load_audio(ref_audio)
        ref_mel = extract_mel_spectrogram(waveform, sr)

        return {
            "text_input_ids": text_tokens["input_ids"].squeeze(0),
            "text_attention_mask": text_tokens["attention_mask"].squeeze(0),
            "audio_codes": audio_codes,
            "ref_mel": torch.from_numpy(ref_mel),
            "language": language
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples."""
    # Find max lengths
    max_text_len = max(item["text_input_ids"].shape[0] for item in batch)
    max_codec_len = max(item["audio_codes"].shape[0] for item in batch)
    max_mel_len = max(item["ref_mel"].shape[1] for item in batch)
    n_mels = batch[0]["ref_mel"].shape[0]

    batch_size = len(batch)

    # Initialize tensors with padding
    text_input_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    text_attention_mask = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    audio_codes = torch.zeros(batch_size, max_codec_len, batch[0]["audio_codes"].shape[-1] if batch[0]["audio_codes"].dim() > 1 else 16, dtype=torch.long)
    ref_mels = torch.zeros(batch_size, n_mels, max_mel_len)
    codec_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        # Text
        text_len = item["text_input_ids"].shape[0]
        text_input_ids[i, :text_len] = item["text_input_ids"]
        text_attention_mask[i, :text_len] = item["text_attention_mask"]

        # Audio codes
        codec_len = item["audio_codes"].shape[0]
        if item["audio_codes"].dim() == 1:
            # Expand to [time, 16] if needed
            audio_codes[i, :codec_len, 0] = item["audio_codes"]
        else:
            audio_codes[i, :codec_len] = item["audio_codes"]
        codec_lengths[i] = codec_len

        # Reference mel
        mel_len = item["ref_mel"].shape[1]
        ref_mels[i, :, :mel_len] = item["ref_mel"]

    return {
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "audio_codes": audio_codes,
        "ref_mels": ref_mels,
        "codec_lengths": codec_lengths
    }


def load_jsonl_data(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
