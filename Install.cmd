@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m llmopenchat install %*
if errorlevel 1 goto failed
echo Installation complete. Open Start-Chat.cmd to chat.
pause
exit /b 0
:failed
echo Installation failed. See the message above. Re-running resumes model download.
pause
exit /b 1
