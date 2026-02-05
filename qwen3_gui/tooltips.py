"""
ELI5 (Explain Like I'm 5) tooltips for Qwen3-TTS GUI.

All tooltips are written to be understandable by someone with
no machine learning background.
"""

from PySide6.QtWidgets import QWidget

from .translations import tr

# ---------------------------------------------------------------------------
# Tooltip key mapping
# ---------------------------------------------------------------------------

TOOLTIP_KEYS = {
    "model": "tooltip_model",
    "language": "tooltip_language",
    "speaker": "tooltip_speaker",
    "text_prompt": "tooltip_text_prompt",
    "instruction": "tooltip_instruction",
    "ref_audio": "tooltip_ref_audio",
    "ref_text": "tooltip_ref_text",
    "xvector": "tooltip_xvector",
    "temperature": "tooltip_temperature",
    "top_k": "tooltip_top_k",
    "top_p": "tooltip_top_p",
    "rep_penalty": "tooltip_rep_penalty",
    "max_tokens": "tooltip_max_tokens",
    "dtype": "tooltip_dtype",
    "flash_attn": "tooltip_flash_attn",
    "epochs": "tooltip_epochs",
    "learning_rate": "tooltip_learning_rate",
    "batch_size": "tooltip_batch_size",
}


def set_tooltip(widget: QWidget, key: str) -> None:
    """Set an ELI5 tooltip on a widget."""
    if key in TOOLTIP_KEYS:
        widget.setToolTip(tr(TOOLTIP_KEYS[key]))
