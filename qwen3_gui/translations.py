"""
Internationalization support for Qwen3-TTS GUI.
"""

from typing import Dict

# Current language (default: English)
_current_language = "en"

# Available languages
LANGUAGES = {
    "en": "English",
    "ru": "Русский",
}

# Translation strings
_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # App
    "app_title": {
        "en": "Qwen3-TTS GUI",
        "ru": "Qwen3-TTS Интерфейс",
    },

    # Menu
    "menu_file": {
        "en": "&File",
        "ru": "&Файл",
    },
    "menu_open_output": {
        "en": "Open Output Folder",
        "ru": "Открыть папку вывода",
    },
    "menu_exit": {
        "en": "E&xit",
        "ru": "В&ыход",
    },
    "menu_help": {
        "en": "&Help",
        "ru": "&Помощь",
    },
    "menu_check_updates": {
        "en": "Check for Updates",
        "ru": "Проверить обновления",
    },
    "menu_about": {
        "en": "&About",
        "ru": "&О программе",
    },

    # Tabs
    "tab_tts": {
        "en": "Text-to-Speech",
        "ru": "Текст в речь",
    },
    "tab_dataset": {
        "en": "Dataset Builder",
        "ru": "Создание датасета",
    },
    "tab_training": {
        "en": "Voice Training",
        "ru": "Обучение голоса",
    },

    # TTS Tab
    "model_selection": {
        "en": "Model Selection",
        "ru": "Выбор модели",
    },
    "language": {
        "en": "Language",
        "ru": "Язык",
    },
    "speaker": {
        "en": "Speaker",
        "ru": "Диктор",
    },
    "text_to_speak": {
        "en": "Text to Speak",
        "ru": "Текст для озвучки",
    },
    "text_placeholder": {
        "en": "Enter the text you want the AI to speak...",
        "ru": "Введите текст, который должен озвучить ИИ...",
    },
    "instruction_optional": {
        "en": "Instruction (optional)",
        "ru": "Инструкция (опционально)",
    },
    "instruction_placeholder": {
        "en": "e.g., 'Speak happily' or 'Sound tired'...",
        "ru": "напр., 'Говори радостно' или 'Звучи устало'...",
    },
    "voice_description_required": {
        "en": "Voice Description (required)",
        "ru": "Описание голоса (обязательно)",
    },
    "reference_audio": {
        "en": "Reference Audio (for cloning)",
        "ru": "Референсное аудио (для клонирования)",
    },
    "ref_audio_placeholder": {
        "en": "Path to reference audio file...",
        "ru": "Путь к референсному аудиофайлу...",
    },
    "reference_transcript": {
        "en": "Reference Transcript:",
        "ru": "Транскрипт референса:",
    },
    "ref_transcript_placeholder": {
        "en": "What's being said in the reference audio...",
        "ru": "Что говорится в референсном аудио...",
    },
    "xvector_mode": {
        "en": "X-vector only mode (no transcript needed)",
        "ru": "Режим x-vector (транскрипт не нужен)",
    },
    "advanced_options": {
        "en": "Advanced Options",
        "ru": "Расширенные настройки",
    },
    "temperature": {
        "en": "Temperature:",
        "ru": "Температура:",
    },
    "top_k": {
        "en": "Top-K:",
        "ru": "Top-K:",
    },
    "top_p": {
        "en": "Top-P:",
        "ru": "Top-P:",
    },
    "repetition_penalty": {
        "en": "Repetition Penalty:",
        "ru": "Штраф повторения:",
    },
    "max_tokens": {
        "en": "Max Tokens:",
        "ru": "Макс. токенов:",
    },
    "dtype": {
        "en": "Dtype:",
        "ru": "Тип данных:",
    },
    "flash_attention": {
        "en": "Use Flash Attention 2",
        "ru": "Использовать Flash Attention 2",
    },
    "output": {
        "en": "Output",
        "ru": "Вывод",
    },
    "browse": {
        "en": "Browse...",
        "ru": "Обзор...",
    },
    "generate_speech": {
        "en": "Generate Speech",
        "ru": "Сгенерировать речь",
    },
    "cancel": {
        "en": "Cancel",
        "ru": "Отмена",
    },
    "ready_device": {
        "en": "Ready. Device:",
        "ru": "Готов. Устройство:",
    },
    "cancelling": {
        "en": "Cancelling...",
        "ru": "Отмена...",
    },

    # Validation errors
    "error_no_text": {
        "en": "Please enter some text to speak.",
        "ru": "Пожалуйста, введите текст для озвучки.",
    },
    "error_no_voice_desc": {
        "en": "Voice description is required for Voice Design mode.",
        "ru": "Описание голоса обязательно для режима Voice Design.",
    },
    "error_no_ref_audio": {
        "en": "Reference audio file is required for Voice Clone mode.",
        "ru": "Референсное аудио обязательно для режима Voice Clone.",
    },
    "error_ref_not_found": {
        "en": "Reference audio file not found:",
        "ru": "Референсное аудио не найдено:",
    },
    "error_no_transcript": {
        "en": "Reference transcript is required (or enable X-vector only mode).",
        "ru": "Транскрипт обязателен (или включите режим x-vector).",
    },
    "error_no_output": {
        "en": "Please specify an output file path.",
        "ru": "Пожалуйста, укажите путь для сохранения.",
    },
    "validation_error": {
        "en": "Validation Error",
        "ru": "Ошибка валидации",
    },

    # Dataset Builder
    "dataset_builder_title": {
        "en": "Dataset Builder",
        "ru": "Создание датасета",
    },
    "dataset_builder_info": {
        "en": "Build a dataset for voice training by matching audio files with their transcripts.\nYou can import audio files and manually transcribe them, or import existing transcript files.",
        "ru": "Создайте датасет для обучения голоса, сопоставив аудиофайлы с их транскриптами.\nВы можете импортировать аудиофайлы и транскрибировать их вручную, или импортировать готовые файлы транскриптов.",
    },
    "import_audio_files": {
        "en": "Import Audio Files...",
        "ru": "Импорт аудиофайлов...",
    },
    "import_transcript": {
        "en": "Import Transcript File...",
        "ru": "Импорт транскрипта...",
    },
    "import_folder": {
        "en": "Import Audio Folder...",
        "ru": "Импорт папки с аудио...",
    },
    "col_audio_file": {
        "en": "Audio File",
        "ru": "Аудиофайл",
    },
    "col_duration": {
        "en": "Duration",
        "ru": "Длительность",
    },
    "col_transcript": {
        "en": "Transcript",
        "ru": "Транскрипт",
    },
    "col_actions": {
        "en": "Actions",
        "ru": "Действия",
    },
    "dataset_info": {
        "en": "Dataset: {count} entries, {duration} total duration",
        "ru": "Датасет: {count} записей, {duration} общая длительность",
    },
    "dataset_name": {
        "en": "Dataset Name:",
        "ru": "Имя датасета:",
    },
    "save_dataset": {
        "en": "Save Dataset",
        "ru": "Сохранить датасет",
    },
    "clear_all": {
        "en": "Clear All",
        "ru": "Очистить всё",
    },
    "play": {
        "en": "Play",
        "ru": "Воспр.",
    },
    "dataset_saved": {
        "en": "Dataset Saved",
        "ru": "Датасет сохранён",
    },
    "dataset_saved_to": {
        "en": "Dataset saved to:",
        "ru": "Датасет сохранён в:",
    },
    "clear_dataset": {
        "en": "Clear Dataset",
        "ru": "Очистить датасет",
    },
    "clear_confirm": {
        "en": "Are you sure you want to clear all entries?",
        "ru": "Вы уверены, что хотите очистить все записи?",
    },
    "import_error": {
        "en": "Import Error",
        "ru": "Ошибка импорта",
    },

    # Training Tab
    "training_title": {
        "en": "Voice Model Training",
        "ru": "Обучение голосовой модели",
    },
    "training_info": {
        "en": "Train a custom voice model using your dataset.\nNote: Training requires significant GPU memory and time.",
        "ru": "Обучите собственную голосовую модель на вашем датасете.\nПримечание: Обучение требует значительных ресурсов GPU и времени.",
    },
    "dataset": {
        "en": "Dataset",
        "ru": "Датасет",
    },
    "select_dataset": {
        "en": "Select Dataset:",
        "ru": "Выберите датасет:",
    },
    "refresh": {
        "en": "Refresh",
        "ru": "Обновить",
    },
    "training_params": {
        "en": "Training Parameters",
        "ru": "Параметры обучения",
    },
    "base_model": {
        "en": "Base Model:",
        "ru": "Базовая модель:",
    },
    "epochs": {
        "en": "Epochs:",
        "ru": "Эпохи:",
    },
    "learning_rate": {
        "en": "Learning Rate:",
        "ru": "Скорость обучения:",
    },
    "batch_size": {
        "en": "Batch Size:",
        "ru": "Размер батча:",
    },
    "output_model_name": {
        "en": "Output Model Name:",
        "ru": "Имя выходной модели:",
    },
    "start_training": {
        "en": "Start Training",
        "ru": "Начать обучение",
    },
    "stop_training": {
        "en": "Stop Training",
        "ru": "Остановить обучение",
    },
    "training_log": {
        "en": "Training Log",
        "ru": "Лог обучения",
    },
    "no_dataset": {
        "en": "No Dataset",
        "ru": "Нет датасета",
    },
    "no_dataset_msg": {
        "en": "Please select or create a dataset first.",
        "ru": "Сначала выберите или создайте датасет.",
    },
    "no_model_name": {
        "en": "No Model Name",
        "ru": "Нет имени модели",
    },
    "no_model_name_msg": {
        "en": "Please enter a name for the output model.",
        "ru": "Пожалуйста, введите имя для выходной модели.",
    },
    "training_placeholder": {
        "en": "Full training implementation requires additional setup.\nThis feature is a placeholder for the training pipeline.\nSee qwen-tts documentation for training instructions.",
        "ru": "Полная реализация обучения требует дополнительной настройки.\nЭта функция является заглушкой для пайплайна обучения.\nСм. документацию qwen-tts для инструкций по обучению.",
    },

    # Media Player
    "audio_player": {
        "en": "Audio Player",
        "ru": "Аудиоплеер",
    },
    "no_file_loaded": {
        "en": "No file loaded",
        "ru": "Файл не загружен",
    },
    "stop": {
        "en": "Stop",
        "ru": "Стоп",
    },
    "pause": {
        "en": "Pause",
        "ru": "Пауза",
    },
    "vol": {
        "en": "Vol:",
        "ru": "Громк.:",
    },

    # Output Log
    "output_log": {
        "en": "Output Log",
        "ru": "Лог вывода",
    },
    "clear": {
        "en": "Clear",
        "ru": "Очистить",
    },
    "log_initialized": {
        "en": "Qwen3-TTS GUI initialized",
        "ru": "Qwen3-TTS GUI инициализирован",
    },
    "log_ready": {
        "en": "Ready to generate speech",
        "ru": "Готов к генерации речи",
    },

    # About dialog
    "about_title": {
        "en": "About",
        "ru": "О программе",
    },
    "about_text": {
        "en": """<h2>Qwen3-TTS GUI</h2>
<p>Version {version}</p>
<p>GUI for <a href="https://github.com/QwenLM/Qwen3-TTS">Qwen3-TTS</a> by Alibaba Cloud.</p>
<p><a href="https://github.com/{repo}">GitHub</a> · MIT License</p>""",
        "ru": """<h2>Qwen3-TTS GUI</h2>
<p>Версия {version}</p>
<p>Интерфейс для <a href="https://github.com/QwenLM/Qwen3-TTS">Qwen3-TTS</a> от Alibaba Cloud.</p>
<p><a href="https://github.com/{repo}">GitHub</a> · Лицензия MIT</p>""",
    },

    # Update menu
    "menu_auto_update": {
        "en": "Auto-update on Startup",
        "ru": "Автообновление при запуске",
    },
    "update_available": {
        "en": "Update Available",
        "ru": "Доступно обновление",
    },
    "update_message": {
        "en": "A new version ({version}) is available!\nCurrent version: {current}\n\nInstall update now?",
        "ru": "Доступна новая версия ({version})!\nТекущая версия: {current}\n\nУстановить обновление?",
    },
    "update_restart_required": {
        "en": "Update Installed",
        "ru": "Обновление установлено",
    },
    "update_restart_msg": {
        "en": "Update installed successfully!\nPlease restart the application.",
        "ru": "Обновление успешно установлено!\nПожалуйста, перезапустите приложение.",
    },
    "up_to_date": {
        "en": "Up to Date",
        "ru": "Актуальная версия",
    },
    "up_to_date_msg": {
        "en": "You have the latest version ({version}).",
        "ru": "У вас последняя версия ({version}).",
    },
    "update_failed": {
        "en": "Update Failed",
        "ru": "Ошибка обновления",
    },
    "update_failed_msg": {
        "en": "Could not check for updates.",
        "ru": "Не удалось проверить обновления.",
    },
    "checking_updates": {
        "en": "Checking for updates...",
        "ru": "Проверка обновлений...",
    },

    # File dialogs
    "select_ref_audio": {
        "en": "Select Reference Audio",
        "ru": "Выберите референсное аудио",
    },
    "save_output_as": {
        "en": "Save Output As",
        "ru": "Сохранить как",
    },
    "select_audio_files": {
        "en": "Select Audio Files",
        "ru": "Выберите аудиофайлы",
    },
    "select_audio_folder": {
        "en": "Select Audio Folder",
        "ru": "Выберите папку с аудио",
    },
    "select_transcript": {
        "en": "Select Transcript File",
        "ru": "Выберите файл транскрипта",
    },

    # Language selector
    "interface_language": {
        "en": "Language:",
        "ru": "Язык:",
    },

    # -------------------------------------------------------------------------
    # Tooltips
    # -------------------------------------------------------------------------

    "tooltip_model": {
        "en": """<b>Which AI brain to use</b><br><br>
Think of these like different voice actors with different skills:<br><br>
<b>Custom Voice</b> - Uses built-in voice presets (like choosing a character)<br>
<b>Voice Design</b> - You describe what voice you want in plain English<br>
<b>Voice Clone</b> - Copies someone's voice from an audio sample<br><br>
<i>1.7B models sound better but need more computer power.<br>
0.6B models are faster but slightly lower quality.</i>""",
        "ru": """<b>Какой ИИ использовать</b><br><br>
Думайте о них как о разных актёрах озвучки:<br><br>
<b>Custom Voice</b> - Использует встроенные голоса (как выбор персонажа)<br>
<b>Voice Design</b> - Вы описываете желаемый голос словами<br>
<b>Voice Clone</b> - Копирует чей-то голос из аудиозаписи<br><br>
<i>Модели 1.7B звучат лучше, но требуют больше ресурсов.<br>
Модели 0.6B быстрее, но качество чуть ниже.</i>""",
    },

    "tooltip_language": {
        "en": """<b>What language should it speak?</b><br><br>
'Auto' means the AI figures it out automatically from your text.<br><br>
Setting it manually can sometimes sound more natural,<br>
especially for languages that share similar characters.""",
        "ru": """<b>На каком языке говорить?</b><br><br>
'Auto' означает, что ИИ определит язык автоматически по тексту.<br><br>
Ручной выбор иногда звучит естественнее,<br>
особенно для языков с похожими символами.""",
    },

    "tooltip_speaker": {
        "en": """<b>Pick a voice character</b><br><br>
Each speaker has a unique voice personality.<br>
Some are better suited for certain languages<br>
(shown in the description), but all can speak<br>
any supported language.""",
        "ru": """<b>Выберите голос</b><br><br>
Каждый диктор имеет уникальный голос.<br>
Некоторые лучше подходят для определённых языков<br>
(указано в описании), но все могут говорить<br>
на любом поддерживаемом языке.""",
    },

    "tooltip_text_prompt": {
        "en": """<b>What should the voice say?</b><br><br>
Type or paste any text here and the AI will<br>
read it out loud in the selected voice.<br><br>
<i>Tip: Punctuation affects how it sounds!<br>
Periods = pauses, ! = emphasis, ? = rising tone</i>""",
        "ru": """<b>Что должен сказать голос?</b><br><br>
Введите или вставьте текст, и ИИ<br>
прочитает его выбранным голосом.<br><br>
<i>Совет: Пунктуация влияет на звучание!<br>
Точки = паузы, ! = акцент, ? = вопросительная интонация</i>""",
    },

    "tooltip_instruction": {
        "en": """<b>How should they say it?</b><br><br>
Give directions like you're coaching an actor:<br>
- "Speak happily and excited"<br>
- "Sound tired and sleepy"<br>
- "Read this like a news anchor"<br><br>
<i>For Voice Design mode, describe the voice itself:<br>
"Young woman with a warm, friendly tone"</i>""",
        "ru": """<b>Как это произнести?</b><br><br>
Давайте указания как режиссёр актёру:<br>
- "Говори радостно и взволнованно"<br>
- "Звучи устало и сонно"<br>
- "Читай как диктор новостей"<br><br>
<i>Для режима Voice Design опишите сам голос:<br>
"Молодая женщина с тёплым, дружелюбным тоном"</i>""",
    },

    "tooltip_ref_audio": {
        "en": """<b>Voice sample to copy</b><br><br>
Upload a recording of the voice you want to clone.<br><br>
<i>Best results with:</i><br>
- 3-10 seconds of clear speech<br>
- No background music or noise<br>
- Single speaker only""",
        "ru": """<b>Образец голоса для копирования</b><br><br>
Загрузите запись голоса, который хотите клонировать.<br><br>
<i>Лучшие результаты с:</i><br>
- 3-10 секунд чистой речи<br>
- Без фоновой музыки и шума<br>
- Только один говорящий""",
    },

    "tooltip_ref_text": {
        "en": """<b>What's being said in the sample?</b><br><br>
Type exactly what the person says in your<br>
reference audio. This helps the AI understand<br>
how that voice pronounces things.<br><br>
<i>Can skip this if you enable 'x-vector only' mode,<br>
but quality may be lower.</i>""",
        "ru": """<b>Что говорится в образце?</b><br><br>
Напишите точно, что говорит человек в<br>
референсном аудио. Это помогает ИИ понять,<br>
как этот голос произносит слова.<br><br>
<i>Можно пропустить, если включить режим 'x-vector',<br>
но качество может быть ниже.</i>""",
    },

    "tooltip_xvector": {
        "en": """<b>Simple voice copying mode</b><br><br>
When ON: Just copies the general "sound" of the voice<br>
(like the pitch and tone) without needing the transcript.<br><br>
When OFF: Does deeper analysis using the transcript<br>
for more accurate cloning.<br><br>
<i>Try ON if you don't know what's said in the sample.</i>""",
        "ru": """<b>Упрощённый режим копирования</b><br><br>
ВКЛ: Копирует только общее "звучание" голоса<br>
(высоту и тон) без необходимости транскрипта.<br><br>
ВЫКЛ: Делает глубокий анализ с транскриптом<br>
для более точного клонирования.<br><br>
<i>Включите, если не знаете, что говорится в образце.</i>""",
    },

    "tooltip_temperature": {
        "en": """<b>Creativity vs Consistency</b><br><br>
Like a creativity dial for the AI:<br><br>
<b>Low (0.1-0.5):</b> Very consistent, predictable<br>
<b>Medium (0.6-1.0):</b> Natural variation (recommended)<br>
<b>High (1.1+):</b> More expressive but might sound weird<br><br>
<i>Default: 0.9 - a good balance</i>""",
        "ru": """<b>Креативность vs Стабильность</b><br><br>
Как регулятор креативности для ИИ:<br><br>
<b>Низкая (0.1-0.5):</b> Очень стабильно, предсказуемо<br>
<b>Средняя (0.6-1.0):</b> Естественные вариации (рекомендуется)<br>
<b>Высокая (1.1+):</b> Более выразительно, но может звучать странно<br><br>
<i>По умолчанию: 0.9 - хороший баланс</i>""",
    },

    "tooltip_top_k": {
        "en": """<b>Word choice variety</b><br><br>
Limits how many options the AI considers when<br>
deciding what sound comes next.<br><br>
<b>Lower numbers:</b> Safer, more predictable<br>
<b>Higher numbers:</b> More varied, might surprise you<br><br>
<i>Default: 50 - works well for most cases</i>""",
        "ru": """<b>Разнообразие выбора</b><br><br>
Ограничивает количество вариантов, которые ИИ<br>
рассматривает при выборе следующего звука.<br><br>
<b>Меньшие числа:</b> Безопаснее, предсказуемее<br>
<b>Большие числа:</b> Разнообразнее, могут удивить<br><br>
<i>По умолчанию: 50 - подходит для большинства случаев</i>""",
    },

    "tooltip_top_p": {
        "en": """<b>Smart word filtering</b><br><br>
Only keeps the most likely sounds until their<br>
combined probability reaches this threshold.<br><br>
<b>1.0:</b> Use all options (disabled)<br>
<b>0.9:</b> Only top 90% most likely<br><br>
<i>Default: 1.0 - usually best to leave this alone</i>""",
        "ru": """<b>Умная фильтрация</b><br><br>
Оставляет только наиболее вероятные звуки,<br>
пока их суммарная вероятность не достигнет порога.<br><br>
<b>1.0:</b> Использовать все варианты (отключено)<br>
<b>0.9:</b> Только топ 90% вероятных<br><br>
<i>По умолчанию: 1.0 - лучше не трогать</i>""",
    },

    "tooltip_rep_penalty": {
        "en": """<b>Avoid repetitive sounds</b><br><br>
Prevents the AI from getting "stuck" repeating<br>
the same sounds over and over.<br><br>
<b>1.0:</b> No penalty (might loop)<br>
<b>1.05:</b> Light penalty (recommended)<br>
<b>1.5+:</b> Strong penalty (might sound choppy)<br><br>
<i>Default: 1.05</i>""",
        "ru": """<b>Избегать повторений</b><br><br>
Не даёт ИИ "застревать" на повторении<br>
одних и тех же звуков.<br><br>
<b>1.0:</b> Без штрафа (может зациклиться)<br>
<b>1.05:</b> Лёгкий штраф (рекомендуется)<br>
<b>1.5+:</b> Сильный штраф (может звучать рвано)<br><br>
<i>По умолчанию: 1.05</i>""",
    },

    "tooltip_max_tokens": {
        "en": """<b>Maximum audio length</b><br><br>
Limits how long the generated audio can be.<br>
Higher = longer possible output but uses more memory.<br><br>
<i>2048 tokens ~ 2-3 minutes of speech<br>
4096 tokens ~ 5-6 minutes of speech</i>""",
        "ru": """<b>Максимальная длина аудио</b><br><br>
Ограничивает длину генерируемого аудио.<br>
Больше = длиннее результат, но больше памяти.<br><br>
<i>2048 токенов ~ 2-3 минуты речи<br>
4096 токенов ~ 5-6 минут речи</i>""",
    },

    "tooltip_dtype": {
        "en": """<b>Number precision (advanced)</b><br><br>
How precisely the AI does math internally:<br><br>
<b>bfloat16:</b> Fast & efficient (needs newer GPU)<br>
<b>float16:</b> Good for older GPUs<br>
<b>float32:</b> Most compatible but slower<br><br>
<i>If you get errors, try float16 or float32</i>""",
        "ru": """<b>Точность вычислений (продвинутое)</b><br><br>
Насколько точно ИИ делает вычисления:<br><br>
<b>bfloat16:</b> Быстро и эффективно (нужен новый GPU)<br>
<b>float16:</b> Для старых GPU<br>
<b>float32:</b> Максимальная совместимость, но медленнее<br><br>
<i>При ошибках попробуйте float16 или float32</i>""",
    },

    "tooltip_flash_attn": {
        "en": """<b>Speed boost (advanced)</b><br><br>
A faster way to process the AI model.<br>
Needs a newer NVIDIA GPU (RTX 30xx or newer).<br><br>
<i>If it causes errors, the app will automatically<br>
fall back to the normal (slower) method.</i>""",
        "ru": """<b>Ускорение (продвинутое)</b><br><br>
Более быстрый способ обработки модели.<br>
Нужен современный NVIDIA GPU (RTX 30xx или новее).<br><br>
<i>При ошибках приложение автоматически<br>
переключится на обычный (медленный) метод.</i>""",
    },

    "tooltip_epochs": {
        "en": """<b>Training repetitions</b><br><br>
How many times the AI studies your dataset.<br><br>
<b>More epochs:</b> Better learning, takes longer<br>
<b>Fewer epochs:</b> Faster, might not learn well<br><br>
<i>Start with 10-20, increase if results aren't good.</i>""",
        "ru": """<b>Повторения обучения</b><br><br>
Сколько раз ИИ изучает ваш датасет.<br><br>
<b>Больше эпох:</b> Лучше обучение, дольше<br>
<b>Меньше эпох:</b> Быстрее, может не выучить<br><br>
<i>Начните с 10-20, увеличьте если результат плохой.</i>""",
    },

    "tooltip_learning_rate": {
        "en": """<b>Learning speed</b><br><br>
How big of adjustments the AI makes while learning.<br><br>
<b>Too high:</b> Learns fast but might "overshoot"<br>
<b>Too low:</b> Very slow, might get stuck<br><br>
<i>Default: 0.0001 - a safe starting point</i>""",
        "ru": """<b>Скорость обучения</b><br><br>
Насколько большие корректировки делает ИИ при обучении.<br><br>
<b>Слишком высокая:</b> Быстро учится, но может "перелететь"<br>
<b>Слишком низкая:</b> Очень медленно, может застрять<br><br>
<i>По умолчанию: 0.0001 - безопасное начало</i>""",
    },

    "tooltip_batch_size": {
        "en": """<b>Samples per lesson</b><br><br>
How many audio clips the AI looks at simultaneously.<br><br>
<b>Larger:</b> Faster training, needs more GPU memory<br>
<b>Smaller:</b> Slower but works on modest hardware<br><br>
<i>Start with 4, increase if you have lots of VRAM</i>""",
        "ru": """<b>Образцов за урок</b><br><br>
Сколько аудиоклипов ИИ обрабатывает одновременно.<br><br>
<b>Больше:</b> Быстрее обучение, нужно больше видеопамяти<br>
<b>Меньше:</b> Медленнее, но работает на скромном железе<br><br>
<i>Начните с 4, увеличьте если много VRAM</i>""",
    },
}


def get_language() -> str:
    """Get current language code."""
    return _current_language


def set_language(lang_code: str) -> None:
    """Set current language."""
    global _current_language
    if lang_code in LANGUAGES:
        _current_language = lang_code


def tr(key: str, **kwargs) -> str:
    """
    Get translated string for key.

    Args:
        key: Translation key
        **kwargs: Format arguments for the string

    Returns:
        Translated string, or key if not found
    """
    if key not in _TRANSLATIONS:
        return key

    translations = _TRANSLATIONS[key]
    text = translations.get(_current_language, translations.get("en", key))

    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass

    return text
