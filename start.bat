@echo off
echo Stopping any running instances...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8877 "') do taskkill /F /PID %%a 2>nul
timeout /t 1 /nobreak >nul
echo Starting Tarkov Stash Helper (source/dev run)...
cd /d "%~dp0"
python app.py
