@echo off
REM ================================================================
REM DEVELOPMENT MODE - CLEAN START
REM Removes dataset and model for fresh testing
REM Use this to test the complete workflow from scratch
REM ================================================================

echo.
echo ========================================
echo   DEVELOPMENT MODE - CLEAN START
echo ========================================
echo.
echo [CLEANUP] Removing old dataset and model for fresh testing...

REM Delete dataset and model files for clean test
if exist "built-in\dataset\data.pickle" (
    del /f /q "built-in\dataset\data.pickle"
    echo [OK] Removed old dataset
) else (
    echo [INFO] No existing dataset found
)

if exist "built-in\dataset\model.p" (
    del /f /q "built-in\dataset\model.p"
    echo [OK] Removed old model
) else (
    echo [INFO] No existing model found
)

echo.
echo Running launcher_dev.py with Python...
echo Test complete workflow: collect data -^> train model -^> run app
echo.

python launcher_dev.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to run launcher_dev.py
    echo Make sure Python is in your PATH
    pause
)
