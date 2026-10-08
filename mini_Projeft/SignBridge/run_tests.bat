@echo off
REM Runs all backend tests (uses small generated SAMPLE videos, not your dataset).
cd /d "%~dp0backend"
call .venv\Scripts\activate.bat
python -m pytest -q
pause
