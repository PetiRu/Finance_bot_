@echo off
title Crypto ML Trading System Orchestrator
color 0B

echo ====================================================================
echo                   CRYPTO ML TRADING SYSTEM LOADER
echo ====================================================================
echo.
echo Local environment scan initiated...

where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo Error: Python was not found in your system's PATH.
    echo Please install Python 3.9 or higher (https://www.python.org/downloads/)
    echo.
    pause
    exit /b
)

if not exist venv (
    echo.
    echo Virtual environment (venv) not detected. Creating one now...
    python -m venv venv
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Scanning dependencies updates (requirements.txt)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

echo.
echo ====================================================================
echo                   BOOTING SYSTEM ENGINE PIPELINE
echo ====================================================================
echo.
python main.py

echo.
pause
