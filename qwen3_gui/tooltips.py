"""
ELI5 (Explain Like I'm 5) tooltips for Qwen3-TTS GUI.

All tooltips are written to be understandable by someone with
no machine learning background.
"""

from PySide6.QtWidgets import QWidget

# ---------------------------------------------------------------------------
# Tooltip definitions
# ---------------------------------------------------------------------------

TOOLTIPS = {
    # Model selection
    "model": """<b>Which AI brain to use</b><br><br>
Think of these like different voice actors with different skills:<br><br>
<b>Custom Voice</b> - Uses built-in voice presets (like choosing a character)<br>
<b>Voice Design</b> - You describe what voice you want in plain English<br>
<b>Voice Clone</b> - Copies someone's voice from an audio sample<br><br>
<i>1.7B models sound better but need more computer power.<br>
0.6B models are faster but slightly lower quality.</i>""",

    # Language
    "language": """<b>What language should it speak?</b><br><br>
'Auto' means the AI figures it out automatically from your text.<br><br>
Setting it manually can sometimes sound more natural,<br>
especially for languages that share similar characters.""",

    # Speaker
    "speaker": """<b>Pick a voice character</b><br><br>
Each speaker has a unique voice personality.<br>
Some are better suited for certain languages<br>
(shown in the description), but all can speak<br>
any supported language.""",

    # Text prompt
    "text_prompt": """<b>What should the voice say?</b><br><br>
Type or paste any text here and the AI will<br>
read it out loud in the selected voice.<br><br>
<i>Tip: Punctuation affects how it sounds!<br>
Periods = pauses, ! = emphasis, ? = rising tone</i>""",

    # Instruction
    "instruction": """<b>How should they say it?</b><br><br>
Give directions like you're coaching an actor:<br>
- "Speak happily and excited"<br>
- "Sound tired and sleepy"<br>
- "Read this like a news anchor"<br><br>
<i>For Voice Design mode, describe the voice itself:<br>
"Young woman with a warm, friendly tone"</i>""",

    # Reference audio
    "ref_audio": """<b>Voice sample to copy</b><br><br>
Upload a recording of the voice you want to clone.<br><br>
<i>Best results with:</i><br>
- 3-10 seconds of clear speech<br>
- No background music or noise<br>
- Single speaker only""",

    # Reference text
    "ref_text": """<b>What's being said in the sample?</b><br><br>
Type exactly what the person says in your<br>
reference audio. This helps the AI understand<br>
how that voice pronounces things.<br><br>
<i>Can skip this if you enable 'x-vector only' mode,<br>
but quality may be lower.</i>""",

    # X-vector
    "xvector": """<b>Simple voice copying mode</b><br><br>
When ON: Just copies the general "sound" of the voice<br>
(like the pitch and tone) without needing the transcript.<br><br>
When OFF: Does deeper analysis using the transcript<br>
for more accurate cloning.<br><br>
<i>Try ON if you don't know what's said in the sample.</i>""",

    # Temperature
    "temperature": """<b>Creativity vs Consistency</b><br><br>
Like a creativity dial for the AI:<br><br>
<b>Low (0.1-0.5):</b> Very consistent, predictable<br>
<b>Medium (0.6-1.0):</b> Natural variation (recommended)<br>
<b>High (1.1+):</b> More expressive but might sound weird<br><br>
<i>Default: 0.9 - a good balance</i>""",

    # Top-K
    "top_k": """<b>Word choice variety</b><br><br>
Limits how many options the AI considers when<br>
deciding what sound comes next.<br><br>
<b>Lower numbers:</b> Safer, more predictable<br>
<b>Higher numbers:</b> More varied, might surprise you<br><br>
<i>Default: 50 - works well for most cases</i>""",

    # Top-P
    "top_p": """<b>Smart word filtering</b><br><br>
Only keeps the most likely sounds until their<br>
combined probability reaches this threshold.<br><br>
<b>1.0:</b> Use all options (disabled)<br>
<b>0.9:</b> Only top 90% most likely<br><br>
<i>Default: 1.0 - usually best to leave this alone</i>""",

    # Repetition penalty
    "rep_penalty": """<b>Avoid repetitive sounds</b><br><br>
Prevents the AI from getting "stuck" repeating<br>
the same sounds over and over.<br><br>
<b>1.0:</b> No penalty (might loop)<br>
<b>1.05:</b> Light penalty (recommended)<br>
<b>1.5+:</b> Strong penalty (might sound choppy)<br><br>
<i>Default: 1.05</i>""",

    # Max tokens
    "max_tokens": """<b>Maximum audio length</b><br><br>
Limits how long the generated audio can be.<br>
Higher = longer possible output but uses more memory.<br><br>
<i>2048 tokens ~ 2-3 minutes of speech<br>
4096 tokens ~ 5-6 minutes of speech</i>""",

    # Dtype
    "dtype": """<b>Number precision (advanced)</b><br><br>
How precisely the AI does math internally:<br><br>
<b>bfloat16:</b> Fast & efficient (needs newer GPU)<br>
<b>float16:</b> Good for older GPUs<br>
<b>float32:</b> Most compatible but slower<br><br>
<i>If you get errors, try float16 or float32</i>""",

    # Flash attention
    "flash_attn": """<b>Speed boost (advanced)</b><br><br>
A faster way to process the AI model.<br>
Needs a newer NVIDIA GPU (RTX 30xx or newer).<br><br>
<i>If it causes errors, the app will automatically<br>
fall back to the normal (slower) method.</i>""",

    # Dataset - audio folder
    "dataset_audio": """<b>Folder with voice recordings</b><br><br>
Point this to a folder containing audio files<br>
(.wav, .mp3, .flac) of the voice you want to train.<br><br>
<i>Best results with:</i><br>
- 30+ minutes of audio total<br>
- Clear speech, minimal background noise<br>
- Consistent recording quality""",

    # Dataset - transcript
    "dataset_transcript": """<b>Text file with what's said</b><br><br>
A text file matching audio to transcripts.<br><br>
Format: <code>filename|transcript text</code><br>
Example: <code>audio001.wav|Hello, how are you?</code><br><br>
<i>One line per audio file.</i>""",

    # Training epochs
    "epochs": """<b>Training repetitions</b><br><br>
How many times the AI studies your dataset.<br><br>
<b>More epochs:</b> Better learning, takes longer<br>
<b>Fewer epochs:</b> Faster, might not learn well<br><br>
<i>Start with 10-20, increase if results aren't good.</i>""",

    # Learning rate
    "learning_rate": """<b>Learning speed</b><br><br>
How big of adjustments the AI makes while learning.<br><br>
<b>Too high:</b> Learns fast but might "overshoot"<br>
<b>Too low:</b> Very slow, might get stuck<br><br>
<i>Default: 0.0001 - a safe starting point</i>""",

    # Batch size
    "batch_size": """<b>Samples per lesson</b><br><br>
How many audio clips the AI looks at simultaneously.<br><br>
<b>Larger:</b> Faster training, needs more GPU memory<br>
<b>Smaller:</b> Slower but works on modest hardware<br><br>
<i>Start with 4, increase if you have lots of VRAM</i>""",
}


def set_tooltip(widget: QWidget, key: str) -> None:
    """Set an ELI5 tooltip on a widget."""
    if key in TOOLTIPS:
        widget.setToolTip(TOOLTIPS[key])
