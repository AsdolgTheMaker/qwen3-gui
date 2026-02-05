# Qwen3-TTS GUI

A modern interface for [Qwen3 Text-to-Speech](https://github.com/QwenLM/Qwen3-TTS).

## Features

- **Text-to-Speech**: Custom voices, voice design from description, voice cloning
- **Dataset Builder**: Create training datasets from audio files
- **Voice Training**: Fine-tune models (placeholder)
- **Multi-language**: English/Russian interface, 10 TTS languages supported

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (8GB+ VRAM for 1.7B, 4GB+ for 0.6B)

## Quick Start

```bash
python run.py
```

The launcher auto-installs dependencies and checks for updates.

## Voice Modes

| Mode | Description |
|------|-------------|
| Custom Voice | 9 built-in speakers with style instructions |
| Voice Design | Describe any voice in natural language |
| Voice Clone | Clone from a 3-10 second audio sample |

## License

MIT
