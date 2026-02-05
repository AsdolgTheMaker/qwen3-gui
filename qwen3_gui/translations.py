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
        "en": "About Qwen3-TTS GUI",
        "ru": "О программе Qwen3-TTS GUI",
    },
    "about_description": {
        "en": "A modern interface for Qwen3 Text-to-Speech.",
        "ru": "Современный интерфейс для Qwen3 Text-to-Speech.",
    },
    "about_features": {
        "en": "Features:",
        "ru": "Возможности:",
    },
    "about_feature_1": {
        "en": "Multiple voice modes (Custom, Design, Clone)",
        "ru": "Различные режимы голоса (Custom, Design, Clone)",
    },
    "about_feature_2": {
        "en": "Built-in media player",
        "ru": "Встроенный медиаплеер",
    },
    "about_feature_3": {
        "en": "Dataset builder",
        "ru": "Создание датасетов",
    },
    "about_feature_4": {
        "en": "Voice training tools",
        "ru": "Инструменты обучения голоса",
    },

    # Update dialog
    "update_available": {
        "en": "Update Available",
        "ru": "Доступно обновление",
    },
    "update_message": {
        "en": "A new version ({version}) is available!\nCurrent version: {current}\n\nWould you like to open the download page?",
        "ru": "Доступна новая версия ({version})!\nТекущая версия: {current}\n\nОткрыть страницу загрузки?",
    },
    "up_to_date": {
        "en": "Up to Date",
        "ru": "Актуальная версия",
    },
    "up_to_date_msg": {
        "en": "You have the latest version!",
        "ru": "У вас последняя версия!",
    },
    "update_failed": {
        "en": "Update Check Failed",
        "ru": "Ошибка проверки обновлений",
    },
    "update_failed_msg": {
        "en": "Could not check for updates.",
        "ru": "Не удалось проверить обновления.",
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
