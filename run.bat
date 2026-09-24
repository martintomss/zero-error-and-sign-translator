@echo off
title Zero-Error Real-Time Hand Sign Translator
color 0b
echo ==============================================================
echo    ZERO-ERROR REAL-TIME HAND SIGN TRANSLATOR (ANTI-GRAVITY AI)
echo ==============================================================
echo.
echo [1/3] Locating Python Environment...

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else if exist "C:\Python311\python.exe" (
    set PYTHON_CMD="C:\Python311\python.exe"
) else (
    set PYTHON_CMD=python
)

echo [2/3] Using Python: %PYTHON_CMD%
echo [3/3] Launching Flask Server on http://localhost:5000...
echo.

start "" "http://localhost:5000"
%PYTHON_CMD% app.py

pause
