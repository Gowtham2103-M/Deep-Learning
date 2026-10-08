@echo off
REM Starts the SignBridge web page (needs the backend running in another window).
cd /d "%~dp0frontend"
if not exist "node_modules" (
  echo Installing packages first time - this takes a minute...
  call npm install
)
call npm run dev
pause
