@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting MosaicDiff Gradio Web Service...
echo Buka browser di http://localhost:7860
echo ===================================================
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" webui.py
) else (
  python webui.py
)
pause
