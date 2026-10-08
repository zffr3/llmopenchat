@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title llmopenchat - Telegram
if not exist ".venv\Scripts\python.exe" (
    echo Please run Install.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m llmopenchat telegram %*
if errorlevel 1 pause
