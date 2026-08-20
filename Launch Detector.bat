@echo off
rem Double-click launcher for the OBI DeepFake Image Detector.
rem
rem Runs the interface straight out of this source checkout, so there is no
rem need to open PowerShell or activate the virtual environment. The console
rem window stays open on purpose: it shows the local URL and any error, and
rem closing it stops the detector.

title OBI DeepFake Image Detector
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo.
    echo   The virtual environment is missing.
    echo   Run setup.ps1 in this folder first, then try again.
    echo.
    pause
    exit /b 1
)

if not exist "models\mobilenet_v2_best.pt" (
    echo.
    echo   The model weights are missing from the models folder.
    echo.
    echo   Download mobilenet_v2_best.pt from the Releases page:
    echo   https://github.com/ObaidROfficial/deepfake-detection-msc/releases/latest
    echo   and put it in the models folder.
    echo.
    pause
    exit /b 1
)

echo.
echo   Starting the OBI DeepFake Image Detector...
echo   The window opens in a few seconds while the model loads.
echo.
echo   This console stays open behind the app.
echo   Closing it, or pressing Ctrl+C, stops the detector.
echo.

"venv\Scripts\python.exe" app.py %*

echo.
echo   The detector has stopped.
pause
