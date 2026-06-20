@echo off
title Voice Emotion Detector AI

echo ======================================================
echo Starting Voice Emotion Detector AI Local Server...
echo ======================================================

:: Activate root virtual environment
call "%~dp0venv\Scripts\activate.bat"

:: Open the default browser to the local site
start http://127.0.0.1:5000

:: Run the Flask server in this console
python backend/app.py

pause
