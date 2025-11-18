@echo off
REM SwimVision Pro - Windows Setup Script

echo =========================================
echo SwimVision Pro - Setup Script (Windows)
echo =========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    exit /b 1
)
echo Python detected
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install dependencies
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt -r requirements-dev.txt
echo.

REM Install package
echo Installing SwimVision in editable mode...
pip install -e .
echo.

REM Set up pre-commit
echo Setting up pre-commit hooks...
pre-commit install
echo.

REM Create .env
echo Setting up environment variables...
if not exist ".env" (
    copy .env.example .env
    echo .env file created from .env.example
    echo Please update .env with your settings
) else (
    echo .env file already exists
)
echo.

REM Create directories
echo Creating necessary directories...
mkdir data\videos 2>nul
mkdir data\exports 2>nul
mkdir models\pose_models 2>nul
mkdir models\ideal_techniques 2>nul
mkdir logs 2>nul
echo.

echo =========================================
echo Setup complete!
echo =========================================
echo.
echo Next steps:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Update .env with your settings
echo   3. Run tests: pytest tests/
echo   4. Start the app: streamlit run app.py
echo.

pause
