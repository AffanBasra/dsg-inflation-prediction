@echo off
echo ═══════════════════════════════════════════════════════════
echo   DSG Inflation Forecasting — Run
echo ═══════════════════════════════════════════════════════════
echo.
echo   Launching 1 Master + 4 Worker JVMs...
echo.

java -cp bin dsg.IntegrationLauncher

echo.
echo   Done.
