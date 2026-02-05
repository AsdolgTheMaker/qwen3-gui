"""
Training loop for Qwen3-TTS fine-tuning.

Based on the official Qwen3-TTS finetuning implementation.
"""

import json
import shutil
from pathlib import Path
from typing import Callable, Optional, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .dataset import TTSDataset, load_jsonl_data


# Global variable to store target speaker embedding
target_speaker_embedding = None


def run_training(
    train_jsonl: Path,
    output_dir: Path,
    speaker_name: str,
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 4,
    device: str = "cuda:0",
    progress_callback: Optional[Callable[[str], None]] = None,
    epoch_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Optional[Path]:
    """
    Run fine-tuning on prepared training data.

    Args:
        train_jsonl: Path to prepared training JSONL (with audio_codes)
        output_dir: Directory to save checkpoints
        speaker_name: Name for the custom speaker
        base_model: Base model to fine-tune
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Number of gradient accumulation steps
        device: Device to train on
        progress_callback: Callback for progress messages
        epoch_callback: Callback with (epoch, total_epochs, loss)
        cancel_check: Function that returns True if training should stop

    Returns:
        Path to final checkpoint, or None if cancelled/failed
    """
    global target_speaker_embedding

    from qwen_tts import Qwen3TTSModel
    from transformers import AutoConfig, AutoProcessor

    if progress_callback:
        progress_callback(f"Loading base model: {base_model}...")

    # Load model
    try:
        qwen3tts = Qwen3TTSModel.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True
        )
    except Exception:
        qwen3tts = Qwen3TTSModel.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map=device
        )

    # Load config
    try:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    except Exception:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # Get the underlying model for training
    model = qwen3tts.model
    model.train()

    if progress_callback:
        progress_callback("Loading training data...")

    # Load training data
    data_list = load_jsonl_data(train_jsonl)
    dataset = TTSDataset(data_list, processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )

    if progress_callback:
        progress_callback(f"Training on {len(dataset)} samples, {len(dataloader)} batches per epoch")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training loop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_checkpoint = None
    accumulation_step = 0

    for epoch in range(num_epochs):
        if cancel_check and cancel_check():
            if progress_callback:
                progress_callback("Training cancelled")
            return None

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if cancel_check and cancel_check():
                if progress_callback:
                    progress_callback("Training cancelled")
                return None

            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            codec_ids = batch['codec_ids'].to(device)
            ref_mels = batch['ref_mels'].to(device).to(torch.bfloat16)
            text_embedding_mask = batch['text_embedding_mask'].to(device)
            codec_embedding_mask = batch['codec_embedding_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            codec_0_labels = batch['codec_0_labels'].to(device)
            codec_mask = batch['codec_mask'].to(device)

            # Get speaker embedding (detached - not trained)
            speaker_embedding = model.speaker_encoder(ref_mels).detach()
            if target_speaker_embedding is None:
                target_speaker_embedding = speaker_embedding.clone()

            # Get text and codec input IDs
            input_text_ids = input_ids[:, :, 0]
            input_codec_ids = input_ids[:, :, 1]

            # Build input embeddings
            input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
            input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

            # Inject speaker embedding at position 6
            input_codec_embedding[:, 6, :] = speaker_embedding

            # Combine text and codec embeddings
            input_embeddings = input_text_embedding + input_codec_embedding

            # Add embeddings for codec channels 1-15
            for i in range(1, 16):
                codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                input_embeddings = input_embeddings + codec_i_embedding

            # Forward pass through talker (without last token, labels shifted by 1)
            outputs = model.talker(
                inputs_embeds=input_embeddings[:, :-1, :],
                attention_mask=attention_mask[:, :-1],
                labels=codec_0_labels[:, 1:],
                output_hidden_states=True
            )

            # Get hidden states for sub-talker finetuning
            hidden_states = outputs.hidden_states[0][-1]
            talker_hidden_states = hidden_states[codec_mask[:, 1:]]
            talker_codec_ids = codec_ids[codec_mask]

            # Sub-talker finetune for codec channels 1-15
            _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                talker_codec_ids, talker_hidden_states
            )

            # Combined loss
            loss = outputs.loss + 0.3 * sub_talker_loss

            # Backward pass with gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_step += 1

            if accumulation_step >= gradient_accumulation_steps:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_step = 0

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                if progress_callback:
                    progress_callback(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # Handle remaining gradients
        if accumulation_step > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accumulation_step = 0

        avg_loss = epoch_loss / max(num_batches, 1)

        if epoch_callback:
            epoch_callback(epoch + 1, num_epochs, avg_loss)

        if progress_callback:
            progress_callback(f"Epoch {epoch+1}/{num_epochs} complete, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        _save_checkpoint(model, base_model, checkpoint_dir, speaker_name, target_speaker_embedding)
        final_checkpoint = checkpoint_dir

        if progress_callback:
            progress_callback(f"Saved checkpoint: {checkpoint_dir}")

    if progress_callback:
        progress_callback(f"Training complete! Final checkpoint: {final_checkpoint}")

    return final_checkpoint


def _save_checkpoint(
    model: Any,
    base_model: str,
    checkpoint_dir: Path,
    speaker_name: str,
    speaker_embedding: torch.Tensor
):
    """Save a training checkpoint compatible with Qwen3TTSModel.from_pretrained()."""
    from safetensors.torch import save_file
    from huggingface_hub import snapshot_download
    import os

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy base model files
    try:
        if os.path.isdir(base_model):
            base_path = Path(base_model)
        else:
            base_path = Path(snapshot_download(base_model))
        shutil.copytree(base_path, checkpoint_dir, dirs_exist_ok=True)
    except Exception:
        pass

    # Update config for custom voice
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {speaker_name: 3000}
        talker_config["spk_is_dialect"] = {speaker_name: False}
        config_dict["talker_config"] = talker_config

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Save model weights
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Remove speaker encoder weights (not finetuned)
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]

    # Inject speaker embedding into codec embedding at position 3000
    if speaker_embedding is not None and 'talker.model.codec_embedding.weight' in state_dict:
        weight = state_dict['talker.model.codec_embedding.weight']
        state_dict['talker.model.codec_embedding.weight'][3000] = speaker_embedding[0].detach().cpu().to(weight.dtype)

    # Save as safetensors
    save_file(state_dict, checkpoint_dir / "model.safetensors")
