@echo off
title Can You Run AI Launcher
cd /d "%~dp0"

set "EXE=%~dp0node_modules\electron\dist\electron.exe"
set "APP=%~dp0"

if not exist "%EXE%" (
  echo.
  echo ERROR: Electron not found at:
  echo   %EXE%
  echo.
  echo Run this first if you have Node:
  echo   npm install
  echo.
  pause
  exit /b 1
)

echo.
echo Starting Can You Run AI...
echo EXE: %EXE%
echo APP: %APP%
echo.

REM Kill old instances so we don't open a zombie process
taskkill /F /IM electron.exe >nul 2>&1

start "Can You Run AI" "%EXE%" "%APP%"

echo.
echo Window should open now: "Can You Run AI"
echo If not, check the Windows taskbar.
echo.
timeout /t 4 >nul
