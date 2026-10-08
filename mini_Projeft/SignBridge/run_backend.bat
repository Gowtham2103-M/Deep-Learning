@echo off
REM Starts the SignBridge backend. Double-click or run from the project folder.
cd /d "%~dp0backend"
if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Follow README section 2 first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload --port 8000
pause
