@echo off
REM Phase 3 Setup Script for Windows
REM Run this to set up your environment

echo ======================================================================
echo   Phase 3: Buy vs. Rent RL - Setup Script (Windows)
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo √ Python found
python --version
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo √ pip found
pip --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo √ Virtual environment created and activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

echo.

REM Install requirements
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ======================================================================
echo √ Setup complete!
echo ======================================================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate when done, run:
echo   deactivate
echo.
echo Next steps:
echo   1. Copy calibrated_params.py from phase2: copy ..\phase2\calibrated_params.py .
echo   2. Run the demo notebook: jupyter notebook phase3_demo.ipynb
echo   3. Or run scripts directly:
echo      - python buy_rent_environment.py
echo      - python modified_dqn.py
echo      - python evaluate_policies.py
echo.
echo ======================================================================
pause
