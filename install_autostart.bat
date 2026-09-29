@echo off
chcp 65001 >nul
cd /d "%~dp0"
schtasks /Create /TN "CryptoRadar" /TR "\"%~dp0start_monitor.bat\"" /SC ONLOGON /RL LIMITED /F
if %errorlevel%==0 (echo Registered: CryptoRadar will start automatically when you log in to Windows.) else (echo Failed. Try right-click - Run as administrator.)
echo Tip: Settings - System - Power - set "Sleep" to Never, otherwise monitoring stops when the PC sleeps.
pause
