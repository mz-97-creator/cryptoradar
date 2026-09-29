@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] Creating virtual environment .venv ...
python -m venv .venv || (echo Python not found. Install Python 3.11+ from python.org and tick "Add to PATH". & pause & exit /b 1)
echo [2/3] Installing packages ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo [3/3] Preparing config.yaml ...
if not exist config.yaml copy config.example.yaml config.yaml
echo.
echo Done. Next: open config.yaml with Notepad and fill in pushplus_token,
echo then double-click test_push.bat
pause
