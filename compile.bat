@echo off
echo ═══════════════════════════════════════════════════════════
echo   DSG Inflation Forecasting — Compile
echo ═══════════════════════════════════════════════════════════
echo.

if not exist bin mkdir bin

javac -d bin src\dsg\*.java

if %ERRORLEVEL% EQU 0 (
    echo.
    echo   [OK] Compilation successful.
    echo   Output directory: bin\dsg\
) else (
    echo.
    echo   [FAIL] Compilation failed. Check errors above.
)
echo.
