@echo off
rem Builds dist\TarkovStashHelper.exe from TarkovStashHelper.spec in an isolated venv.
rem
rem   build.bat         lean exe: no torch/transformers (what the GitHub release ships)
rem   build.bat dino    EXPERIMENT: also bundle torch + transformers (CPU torch from PyPI) so
rem                     DINOv2 stage 2 works inside the exe; several hundred MB, not released
rem
rem The venv is created once (delete .buildvenv* to rebuild it).  Output: dist\TarkovStashHelper.exe
setlocal
cd /d "%~dp0"

set "TSH_FLAVOUR=%~1"
if "%TSH_FLAVOUR%"=="" set "TSH_FLAVOUR=lean"
if /i "%TSH_FLAVOUR%"=="lean" (
  set "VENV=.buildvenv"
  set "REQ=requirements.txt"
) else if /i "%TSH_FLAVOUR%"=="dino" (
  set "VENV=.buildvenv-dino"
  set "REQ=requirements-dino.txt"
) else (
  echo Unknown flavour "%TSH_FLAVOUR%" - use "lean" or "dino".
  exit /b 2
)

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating isolated build venv %VENV% ...
  python -m venv "%VENV%" || exit /b 1
  "%VENV%\Scripts\python.exe" -m pip install -q --upgrade pip || exit /b 1
  "%VENV%\Scripts\python.exe" -m pip install -q -r "%REQ%" pyinstaller || exit /b 1
)

echo Regenerating app icon...
"%VENV%\Scripts\python.exe" icon_asset.py || exit /b 1

echo Building TarkovStashHelper.exe (%TSH_FLAVOUR%)...
"%VENV%\Scripts\python.exe" -m PyInstaller --noconfirm --clean TarkovStashHelper.spec || exit /b 1

echo.
echo Done. Output: dist\TarkovStashHelper.exe
