# Qwen3-TTS GUI

A modern, user-friendly interface for [Qwen3 Text-to-Speech](https://github.com/QwenLM/Qwen3-TTS).

## Features

- **Text-to-Speech Generation**
  - Custom Voice: Use built-in speaker presets with optional emotion/style instructions
  - Voice Design: Describe any voice in natural language
  - Voice Clone: Clone a voice from a short audio sample

- **Built-in Media Player**
  - Instant playback of generated audio
  - Play/pause, seek, and volume controls

- **Dataset Builder**
  - Import audio files and transcripts
  - Build training datasets with a visual interface
  - Export in standard format for training

- **Voice Training** (placeholder)
  - Fine-tune models on custom datasets
  - Training parameter controls

- **Beginner-Friendly**
  - ELI5 (Explain Like I'm 5) tooltips for all ML concepts
  - No machine learning knowledge required

- **Auto-Update Support**
  - Check for updates from GitHub releases

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ VRAM for 1.7B models, 4GB+ for 0.6B models

## Installation

Simply run the script - it will automatically:
1. Create a virtual environment
2. Install PyTorch with appropriate CUDA support
3. Install all dependencies (PySide6, qwen-tts, etc.)

```bash
python run.py
```

Or on Windows, use the batch file:
```bash
run.bat
```

## Usage

1. **Select a Model**: Choose between Custom Voice, Voice Design, or Voice Clone modes
2. **Enter Text**: Type what you want the AI to say
3. **Configure Settings**: Adjust speaker, language, and advanced parameters
4. **Generate**: Click "Generate Speech" and wait for processing
5. **Play**: Use the built-in player to preview the result

## Voice Modes

### Custom Voice
Use one of 9 built-in speakers with optional style instructions:
- Vivian, Serena, Uncle_Fu (Chinese)
- Dylan (Beijing), Eric (Sichuan)
- Ryan, Aiden (English)
- Ono_Anna (Japanese)
- Sohee (Korean)

### Voice Design
Describe the voice you want in natural language:
- "Young woman with a warm, friendly tone"
- "Deep male voice, authoritative like a news anchor"

### Voice Clone
Provide a short audio sample (3-10 seconds) to clone:
- Upload reference audio
- Optionally provide the transcript for better results
- Use "X-vector only" mode if transcript is unknown

## Supported Languages

Auto-detect or specify: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## License

MIT License

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Cloud
- Built with [PySide6](https://www.qt.io/qt-for-python) (Qt for Python)
