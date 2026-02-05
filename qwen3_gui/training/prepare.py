"""
Data preparation for Qwen3-TTS fine-tuning.

Converts dataset format and tokenizes audio using Qwen3-TTS-Tokenizer.
"""

import json
from pathlib import Path
from typing import Callable, Optional

BATCH_SIZE = 32
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def convert_dataset_to_jsonl(
    dataset_dir: Path,
    output_jsonl: Path,
    ref_audio_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> int:
    """
    Convert transcript.txt format to JSONL format for training.

    Args:
        dataset_dir: Path to dataset folder containing transcript.txt and audio files
        output_jsonl: Path to output JSONL file
        ref_audio_path: Path to reference audio (if None, uses first audio file)
        progress_callback: Optional callback for progress messages

    Returns:
        Number of samples converted
    """
    transcript_file = dataset_dir / "transcript.txt"
    if not transcript_file.exists():
        raise FileNotFoundError(f"transcript.txt not found in {dataset_dir}")

    samples = []
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            audio_name, text = line.split("|", 1)
            audio_path = dataset_dir / audio_name.strip()
            if audio_path.exists():
                samples.append({
                    "audio": str(audio_path),
                    "text": text.strip()
                })

    if not samples:
        raise ValueError("No valid samples found in transcript.txt")

    # Use first audio as reference if not specified
    if ref_audio_path is None:
        ref_audio_path = Path(samples[0]["audio"])

    if progress_callback:
        progress_callback(f"Converting {len(samples)} samples to JSONL format...")

    # Write JSONL with ref_audio field
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            sample["ref_audio"] = str(ref_audio_path)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(samples)


def prepare_training_data(
    input_jsonl: Path,
    output_jsonl: Path,
    device: str = "cuda:0",
    progress_callback: Optional[Callable[[str], None]] = None
) -> int:
    """
    Tokenize audio files and prepare training data with audio codes.

    Args:
        input_jsonl: Path to input JSONL file (with audio, text, ref_audio fields)
        output_jsonl: Path to output JSONL file (with added audio_codes field)
        device: Device to use for tokenization
        progress_callback: Optional callback for progress messages

    Returns:
        Number of samples processed
    """
    from qwen_tts import Qwen3TTSTokenizer

    if progress_callback:
        progress_callback(f"Loading tokenizer ({TOKENIZER_MODEL})...")

    # Load tokenizer with device_map for GPU support
    tokenizer = Qwen3TTSTokenizer.from_pretrained(TOKENIZER_MODEL, device_map=device)

    # Load input data
    data_list = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    if progress_callback:
        progress_callback(f"Tokenizing {len(data_list)} audio files...")

    output_data = []

    # Process one at a time to avoid memory issues and handle errors gracefully
    for idx, item in enumerate(data_list):
        try:
            result = tokenizer.encode([item["audio"]])
            # result.audio_codes is a list of tensors, one per input audio
            # Each tensor has shape [num_codebooks, num_frames]
            audio_codes = result.audio_codes[0]  # Get first (only) result
            enriched = item.copy()
            # Transpose to [frames, codebooks] for easier processing later
            enriched["audio_codes"] = audio_codes.T.cpu().tolist()
            output_data.append(enriched)

            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(f"Tokenized {idx + 1}/{len(data_list)} samples...")

        except Exception as e:
            if progress_callback:
                progress_callback(f"Warning: Failed to tokenize {item['audio']}: {e}")
            continue

    # Write output
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if progress_callback:
        progress_callback(f"Data preparation complete: {len(output_data)} samples")

    return len(output_data)
