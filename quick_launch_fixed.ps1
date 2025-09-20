# AI Trading System - Quick Launch Scripts

Write-Host "AI Trading System - Quick Launch Scripts" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Get the current directory (should be ai_advancements)
$BASE_DIR = Get-Location

# Define paths
$LAUNCHER_SCRIPT = Join-Path $BASE_DIR "daily_launcher.py"

# Check if launcher exists
if (-not (Test-Path $LAUNCHER_SCRIPT)) {
    Write-Host "Error: daily_launcher.py not found in current directory" -ForegroundColor Red
    Write-Host "   Please run this script from the ai_advancements folder" -ForegroundColor Yellow
    exit 1
}

Write-Host "Base Directory: $BASE_DIR" -ForegroundColor Cyan
Write-Host "Launcher Script: Found" -ForegroundColor Green
Write-Host ""

# Menu function
function Show-Menu {
    Write-Host "SELECT OPERATION:" -ForegroundColor Yellow
    Write-Host "=================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Pre-Market Routine (6:00-9:30 AM)" -ForegroundColor White
    Write-Host "   - Database health check" -ForegroundColor Gray
    Write-Host "   - IBKR Gateway connection" -ForegroundColor Gray
    Write-Host "   - Historical data update" -ForegroundColor Gray
    Write-Host "   - IBD 50 stocks loading" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Market Hours Signals (9:30 AM-4:00 PM)" -ForegroundColor White
    Write-Host "   - Generate trading signals" -ForegroundColor Gray
    Write-Host "   - Analyze IBD 50 stocks" -ForegroundColor Gray
    Write-Host "   - Export signal recommendations" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. End-of-Day Analysis (4:00-6:00 PM)" -ForegroundColor White
    Write-Host "   - Final data collection" -ForegroundColor Gray
    Write-Host "   - Performance report" -ForegroundColor Gray
    Write-Host "   - Daily results archive" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Weekend Analysis (Fridays)" -ForegroundColor White
    Write-Host "   - Comprehensive AI analysis" -ForegroundColor Gray
    Write-Host "   - Portfolio optimization" -ForegroundColor Gray
    Write-Host "   - Pattern recognition" -ForegroundColor Gray
    Write-Host ""
    Write-Host "5. Complete Daily Sequence" -ForegroundColor White
    Write-Host "   - All routines in sequence" -ForegroundColor Gray
    Write-Host "   - Full day automation" -ForegroundColor Gray
    Write-Host ""
    Write-Host "6. System Health Check" -ForegroundColor White
    Write-Host "   - Database connection" -ForegroundColor Gray
    Write-Host "   - IBKR Gateway status" -ForegroundColor Gray
    Write-Host "   - Component availability" -ForegroundColor Gray
    Write-Host ""
    Write-Host "0. Exit" -ForegroundColor Red
    Write-Host ""
}

# Main loop
do {
    Show-Menu
    $choice = Read-Host "Enter your choice (0-6)"
    
    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "Starting Pre-Market Routine..." -ForegroundColor Green
            Write-Host "===============================" -ForegroundColor Green
            python $LAUNCHER_SCRIPT --pre-market
            Write-Host ""
            Read-Host "Press Enter to continue..."
        }
        "2" {
            Write-Host ""
            Write-Host "Starting Market Hours Signal Generation..." -ForegroundColor Green
            Write-Host "===========================================" -ForegroundColor Green
            python $LAUNCHER_SCRIPT --market-hours
            Write-Host ""
            Read-Host "Press Enter to continue..."
        }
        "3" {
            Write-Host ""
            Write-Host "Starting End-of-Day Analysis..." -ForegroundColor Green
            Write-Host "===============================" -ForegroundColor Green
            python $LAUNCHER_SCRIPT --end-of-day
            Write-Host ""
            Read-Host "Press Enter to continue..."
        }
        "4" {
            Write-Host ""
            Write-Host "Starting Weekend Analysis..." -ForegroundColor Green
            Write-Host "=============================" -ForegroundColor Green
            Write-Host "WARNING: This may take 30-60 minutes to complete" -ForegroundColor Yellow
            $confirm = Read-Host "Continue? (y/N)"
            if ($confirm -eq "y" -or $confirm -eq "Y") {
                python $LAUNCHER_SCRIPT --weekend
            } else {
                Write-Host "Weekend analysis cancelled." -ForegroundColor Yellow
            }
            Write-Host ""
            Read-Host "Press Enter to continue..."
        }
        "5" {
            Write-Host ""
            Write-Host "Starting Complete Daily Sequence..." -ForegroundColor Green
            Write-Host "===================================" -ForegroundColor Green
            Write-Host "WARNING: This will run all daily routines in sequence" -ForegroundColor Yellow
            Write-Host "   Estimated time: 20-30 minutes" -ForegroundColor Yellow
            $confirm = Read-Host "Continue? (y/N)"
            if ($confirm -eq "y" -or $confirm -eq "Y") {
                python $LAUNCHER_SCRIPT --all
            } else {
                Write-Host "Complete sequence cancelled." -ForegroundColor Yellow
            }
            Write-Host ""
            Read-Host "Press Enter to continue..."
        }
        "6" {
            Write-Host ""
            Write-Host "Running System Health Check..." -ForegroundColor Green
            Write-Host "==============================" -ForegroundColor Green
            python $LAUNCHER_SCRIPT --health-check
            Write-Host ""
            Read-Host "Press Enter to continue..."
        }
        "0" {
            Write-Host ""
            Write-Host "Goodbye! Have a great trading day!" -ForegroundColor Green
            break
        }
        default {
            Write-Host ""
            Write-Host "Invalid choice. Please select 0-6." -ForegroundColor Red
            Write-Host ""
            Start-Sleep -Seconds 2
        }
    }
    
    Clear-Host
    Write-Host "AI Trading System - Quick Launch Scripts" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    
} while ($choice -ne "0")

Write-Host "Quick Launch Scripts - Session Complete" -ForegroundColor Cyan