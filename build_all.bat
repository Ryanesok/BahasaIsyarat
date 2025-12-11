@echo off
REM ================================================================
REM FAST BUILD SCRIPT - Sign Language Reader Distribution
REM Optimized for faster builds with better progress reporting
REM ================================================================

echo.
echo ========================================
echo   SIGN LANGUAGE READER - FAST BUILD
echo ========================================
echo.

REM --- Step 0: Kill any running instances ---
echo [0/5] Stopping running applications...
taskkill /F /IM SignLanguageReader.exe 2>nul
taskkill /F /IM desktop_app.exe 2>nul
taskkill /F /IM collect_data.exe 2>nul
taskkill /F /IM train_model.exe 2>nul
timeout /t 1 /nobreak >nul
echo [OK] Stopped running applications
echo.

REM --- Step 1: Clean previous builds ---
echo [1/5] Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
mkdir dist
mkdir dist\built-in
echo [OK] Clean completed
echo.

REM --- Step 2: Build Launcher (fastest, no heavy dependencies) ---
echo [2/5] Building launcher (SignLanguageReader.exe)...
echo    - This is lightweight, should take ~20 seconds
pyinstaller --clean --noconfirm SignLanguageReader.spec
if errorlevel 1 (
    echo [ERROR] Launcher build failed!
    pause
    exit /b 1
)
echo [OK] Launcher built successfully
echo.

REM --- Step 3: Build Desktop App (heavy - MediaPipe, CV2, sklearn) ---
echo [3/5] Building desktop_app.exe...
echo    - This includes MediaPipe, CV2, and ML libraries
echo    - Expected time: 40-60 seconds
pyinstaller --clean --noconfirm build_desktop_app.spec
if errorlevel 1 (
    echo [ERROR] Desktop app build failed!
    pause
    exit /b 1
)
move /y dist\desktop_app.exe dist\built-in\desktop_app.exe
echo [OK] Desktop app built successfully
echo.

REM --- Step 4: Build Collect Data (heavy - MediaPipe, CV2) ---
echo [4/5] Building collect_data.exe...
echo    - This includes MediaPipe and CV2
echo    - Expected time: 40-60 seconds
pyinstaller --clean --noconfirm build_collect_data.spec
if errorlevel 1 (
    echo [ERROR] Collect data build failed!
    pause
    exit /b 1
)
move /y dist\collect_data.exe dist\built-in\collect_data.exe
echo [OK] Collect data built successfully
echo.

REM --- Step 5: Build Train Model (medium - sklearn only) ---
echo [5/5] Building train_model.exe...
echo    - This includes sklearn only
echo    - Expected time: 20-30 seconds
pyinstaller --clean --noconfirm build_train_model.spec
if errorlevel 1 (
    echo [ERROR] Train model build failed!
    pause
    exit /b 1
)
move /y dist\train_model.exe dist\built-in\train_model.exe
echo [OK] Train model built successfully
echo.

REM --- Copy supporting files ---
echo [6/6] Copying supporting files...
REM Copy only specific files from built-in, not the whole folder
xcopy /y /q "built-in\*.py" "dist\built-in\"
xcopy /y /q "built-in\*.txt" "dist\built-in\"
copy /y config.json dist\config.json
copy /y path_config.py dist\path_config.py
echo [OK] Supporting files copied
echo.

REM --- Create data directories ---
echo Creating data directory structure...
mkdir "dist\sign-language-detector-python\data\static\alphabet" 2>nul
mkdir "dist\sign-language-detector-python\data\static\numbers" 2>nul
mkdir "dist\sign-language-detector-python\data\dynamic\words" 2>nul
mkdir "dist\built-in\dataset" 2>nul
echo [OK] Data directories created
echo.

REM --- Clean up build artifacts ---
echo Cleaning up build artifacts...
rmdir /s /q build
echo [OK] Build artifacts cleaned
echo.

echo ========================================
echo   BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Executables location:
echo   - Launcher:     dist\SignLanguageReader.exe
echo   - Desktop App:  dist\built-in\desktop_app.exe
echo   - Collect Data: dist\built-in\collect_data.exe
echo   - Train Model:  dist\built-in\train_model.exe
echo.
echo You can now test the application by running:
echo   dist\SignLanguageReader.exe
echo.
echo Build time improvements:
echo   - Matplotlib AND scipy kept (required by MediaPipe and sklearn)
echo   - Excluded only: pandas, IPython, jupyter
echo   - train_model now includes cv2/mediapipe for create_dataset execution
echo   - All apps use path_config.py for consistent path resolution
echo   - Direct script execution prevents duplicate windows
echo   - Removed unnecessary notification popups for instant launch
echo.
pause
