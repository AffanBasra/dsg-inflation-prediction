@echo off
echo =========================================
echo  1. Running Distributed Java System
echo =========================================
java -cp bin dsg.IntegrationLauncher

echo.
echo =========================================
echo  2. Generating Plots and Advanced Metrics
echo =========================================
python generate_results.py

echo.
echo Pipeline Complete! Check the output/viz folder.
pause