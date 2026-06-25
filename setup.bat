@echo off
echo ====================================================
echo Starting SkyWrite Clean Project Setup...
echo ====================================================

:: Navigate to the folder where this batch file sits
cd /d "%~dp0"

:: Create a fresh virtual environment
echo Creating virtual environment (sky_env)...
python -m venv sky_env

:: Activate and install only the core running libraries 
echo Installing required libraries (This may take a minute)...
call sky_env\Scripts\activate
pip install opencv-python mediapipe==0.10.11 tensorflow==2.15.0 pillow

:: Lock down the protobuf version match perfectly
echo Locking library version alignments...
pip install protobuf==3.20.3 --force-reinstall

echo ====================================================
echo Setup complete! You can now run the project using Run_AirCanvas.bat
echo ====================================================
pause