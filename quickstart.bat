@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist requirements.txt (
  echo requirements.txt not found.
  goto launch
)

set "missing=0"
for /f "usebackq delims=" %%r in ("requirements.txt") do (
  set "line=%%r"
  if not "!line!"=="" (
    if /i not "!line:~0,1!"=="#" (
      for /f "delims=><=~ " %%p in ("!line!") do (
        python -m pip show "%%p" >nul 2>&1
        if errorlevel 1 set "missing=1"
      )
    )
  )
)

if "!missing!"=="1" (
  echo Installing requirements...
  python -m pip install -r requirements.txt
) else (
  echo Requirements already installed.
)

:launch
echo Launching botmaker...
python botmaker.py

echo.
echo Botmaker exited. Press any key to close.
pause >nul
