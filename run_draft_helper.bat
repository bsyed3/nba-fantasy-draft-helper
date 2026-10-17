@echo off
rem ---------------------------------------------------------------------------
rem  NBA Fantasy Draft Helper - double-click launcher (Windows)
rem
rem  First run : creates a virtual environment and installs requirements.txt
rem              (a few minutes). Later runs start the app in a few seconds.
rem  Then      : starts Streamlit; your browser opens at http://localhost:8501
rem  To stop   : press Ctrl+C in this window, or just close it.
rem
rem  The virtual environment lives in .venv next to this file. To keep it
rem  outside OneDrive, set NBA_VENV first, for example:  set NBA_VENV=C:\nba-venv
rem  Draft state is saved to data\processed\nba.db (override with NBA_DB).
rem ---------------------------------------------------------------------------
setlocal
title NBA Fantasy Draft Helper

cd /d "%~dp0"
if not defined NBA_VENV set "NBA_VENV=%~dp0.venv"
set "VPY=%NBA_VENV%\Scripts\python.exe"

if not exist "%VPY%" (
    echo Creating the virtual environment - first run only ...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%NBA_VENV%"
    ) else (
        python -m venv "%NBA_VENV%"
    )
)
if not exist "%VPY%" (
    echo.
    echo Could not create the virtual environment.
    echo Install Python 3.12 or 3.13 from python.org, tick "Add python.exe to PATH",
    echo then run this file again.
    pause
    exit /b 1
)

"%VPY%" -c "import streamlit, lightgbm, sklearn, scipy, pyarrow, pandas, numpy, nba_api" >nul 2>nul
if errorlevel 1 (
    echo Installing requirements - first run only, this takes a few minutes ...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installing the requirements failed - see the messages above.
        pause
        exit /b 1
    )
)

echo.
echo Starting the draft helper. The first load takes about a minute; a browser tab will open.
echo Leave this window open while you draft.
echo.
"%VPY%" -m streamlit run src\app\streamlit_app.py --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo The app exited with an error - see the messages above.
    pause
)
endlocal
