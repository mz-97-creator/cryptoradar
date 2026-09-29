@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
title CryptoRadar
:loop
.venv\Scripts\python.exe monitor.py
echo CryptoRadar exited, restarting in 60 seconds... (close this window to stop)
timeout /t 60 /nobreak >nul
goto loop
