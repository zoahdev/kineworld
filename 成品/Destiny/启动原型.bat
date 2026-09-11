@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Destiny prototype starting... do not close this window.
node server.js
if errorlevel 1 (
  echo.
  echo Node.js was not found or the server failed to start.
  echo Please install Node.js from https://nodejs.org  then run this file again.
  pause
)
