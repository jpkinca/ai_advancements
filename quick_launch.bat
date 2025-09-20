@echo off
:: AI Trading System - Windows Command Launch Scripts
:: Quick access to daily trading operations

title AI Trading System - Daily Operations

echo.
echo ========================================
echo   AI Trading System - Daily Operations
echo ========================================
echo.

:: Get current directory
set "BASE_DIR=%cd%"
set "LAUNCHER_SCRIPT=%BASE_DIR%\daily_launcher.py"

:: Check if launcher exists
if not exist "%LAUNCHER_SCRIPT%" (
    echo ❌ Error: daily_launcher.py not found
    echo    Please run this script from the ai_advancements folder
    pause
    exit /b 1
)

echo 📂 Base Directory: %BASE_DIR%
echo 🐍 Launcher Script: Found
echo.

:MENU
cls
echo.
echo ========================================
echo   AI Trading System - Daily Operations  
echo ========================================
echo.
echo 🎯 SELECT OPERATION:
echo ===================
echo.
echo 1. 🌅 Pre-Market Routine (6:00-9:30 AM)
echo    - Database health check
echo    - IBKR Gateway connection  
echo    - Historical data update
echo    - IBD 50 stocks loading
echo.
echo 2. 📈 Market Hours Signals (9:30 AM-4:00 PM)
echo    - Generate trading signals
echo    - Analyze IBD 50 stocks
echo    - Export signal recommendations
echo.
echo 3. 🌙 End-of-Day Analysis (4:00-6:00 PM)
echo    - Final data collection
echo    - Performance report
echo    - Daily results archive
echo.
echo 4. 🔬 Weekend Analysis (Fridays)
echo    - Comprehensive AI analysis
echo    - Portfolio optimization  
echo    - Pattern recognition
echo.
echo 5. 🔄 Complete Daily Sequence
echo    - All routines in sequence
echo    - Full day automation
echo.
echo 6. 🏥 System Health Check
echo    - Database connection
echo    - IBKR Gateway status
echo    - Component availability
echo.
echo 0. ❌ Exit
echo.

set /p choice="Enter your choice (0-6): "

if "%choice%"=="1" goto PRE_MARKET
if "%choice%"=="2" goto MARKET_HOURS  
if "%choice%"=="3" goto END_OF_DAY
if "%choice%"=="4" goto WEEKEND
if "%choice%"=="5" goto COMPLETE
if "%choice%"=="6" goto HEALTH_CHECK
if "%choice%"=="0" goto EXIT

echo.
echo ❌ Invalid choice. Please select 0-6.
timeout /t 2 >nul
goto MENU

:PRE_MARKET
echo.
echo ================================
echo 🌅 Starting Pre-Market Routine...
echo ================================
python "%LAUNCHER_SCRIPT%" --pre-market
echo.
pause
goto MENU

:MARKET_HOURS
echo.
echo =============================================  
echo 📈 Starting Market Hours Signal Generation...
echo =============================================
python "%LAUNCHER_SCRIPT%" --market-hours
echo.
pause
goto MENU

:END_OF_DAY
echo.
echo =================================
echo 🌙 Starting End-of-Day Analysis...
echo =================================
python "%LAUNCHER_SCRIPT%" --end-of-day
echo.
pause
goto MENU

:WEEKEND
echo.
echo ===============================
echo 🔬 Starting Weekend Analysis...
echo ===============================
echo ⚠️  This may take 30-60 minutes to complete
set /p confirm="Continue? (y/N): "
if /i "%confirm%"=="y" (
    python "%LAUNCHER_SCRIPT%" --weekend
) else (
    echo Weekend analysis cancelled.
)
echo.
pause
goto MENU

:COMPLETE
echo.
echo =====================================
echo 🔄 Starting Complete Daily Sequence...
echo =====================================
echo ⚠️  This will run all daily routines in sequence
echo    Estimated time: 20-30 minutes
set /p confirm="Continue? (y/N): "
if /i "%confirm%"=="y" (
    python "%LAUNCHER_SCRIPT%" --all
) else (
    echo Complete sequence cancelled.
)
echo.
pause
goto MENU

:HEALTH_CHECK
echo.
echo ================================
echo 🏥 Running System Health Check...
echo ================================
python "%LAUNCHER_SCRIPT%" --health-check
echo.
pause
goto MENU

:EXIT
echo.
echo 👋 Goodbye! Have a great trading day!
echo.
timeout /t 2 >nul
exit /b 0