@echo off
cd /d "%~dp0"
if not exist .buildvenv (
  echo Creating isolated build venv...
  python -m venv .buildvenv
  .buildvenv\Scripts\python.exe -m pip install -q --upgrade pip
  .buildvenv\Scripts\python.exe -m pip install -q -r requirements.txt pyinstaller
)
echo Regenerating app icon...
.buildvenv\Scripts\python.exe icon_asset.py
echo Building TarkovStashHelper.exe...
.buildvenv\Scripts\python.exe -m PyInstaller ^
  --noconfirm ^
  --onefile ^
  --windowed ^
  --name TarkovStashHelper ^
  --icon assets\icon.ico ^
  --add-data "templates;templates" ^
  app.py
echo.
echo Done. Output: dist\TarkovStashHelper.exe
