@echo off
:: Automatically targets whichever folder this batch file sits in
cd /d "%~dp0"

:: Check if the stable setup was run
if not exist sky_env (
    echo Error: Please run setup.bat first to build the environment!
    pause
    exit
)

:: Activate the sandbox and run the app
echo Launching SkyWrite Application...
call sky_env\Scripts\activate
python Final_Code.py
pause