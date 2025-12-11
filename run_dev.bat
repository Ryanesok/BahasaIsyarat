@echo off
REM ================================================================
REM DEVELOPMENT MODE - Test WITHOUT building .exe
REM Use this for FAST TESTING before final build
REM Uses EXISTING dataset and model (normal usage)
REM ================================================================

echo.
echo ========================================
echo   DEVELOPMENT MODE - FAST TESTING
echo ========================================
echo.
echo Running launcher_dev.py with Python...
echo Changes to .py files take effect IMMEDIATELY
echo NO BUILD REQUIRED!
echo.
echo Using EXISTING dataset and model
echo (Use run_dev_clean.bat for fresh start)
echo.

python launcher_dev.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to run launcher_dev.py
    echo Make sure Python is in your PATH
    pause
)
