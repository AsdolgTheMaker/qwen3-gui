@echo off
setlocal

:: -----------------------------------------------------------
:: Qwen3-TTS Launcher
:: Ensures Python is installed, then runs run.py
:: -----------------------------------------------------------

cd /d "%~dp0"

:: 1. Check if Python is available
where python >nul 2>&1
if %errorlevel% equ 0 (
    :: Verify it's a real Python and not the Windows Store stub
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        goto :run
    )
)

:: 2. Check py launcher (installed with official Python)
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [launcher] Using py launcher.
        py -3 run.py
        goto :end
    )
)

:: 3. Python not found — offer to install
echo.
echo  Python is not installed or not on PATH.
echo.
echo  Options:
echo    1. Download from https://www.python.org/downloads/
echo       (make sure to check "Add Python to PATH" during install)
echo    2. Install via winget:
echo       winget install Python.Python.3.12
echo.

choice /c YN /m "Attempt automatic install via winget now? "
if %errorlevel% equ 1 (
    echo.
    echo [launcher] Installing Python 3.12 via winget ...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [launcher] winget install failed. Please install Python manually.
        pause
        goto :end
    )
    echo.
    echo [launcher] Python installed. You may need to restart this terminal
    echo            for PATH changes to take effect.
    echo.
    :: Refresh PATH from registry for the current session
    for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
    for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
    set "PATH=%SYS_PATH%;%USER_PATH%"

    where python >nul 2>&1
    if %errorlevel% equ 0 (
        goto :run
    ) else (
        echo [launcher] Python still not found on PATH. Please restart your terminal and try again.
        pause
        goto :end
    )
) else (
    echo [launcher] Aborted. Please install Python and try again.
    pause
    goto :end
)

:run
echo [launcher] Starting Qwen3-TTS ...
python run.py

:end
endlocal
