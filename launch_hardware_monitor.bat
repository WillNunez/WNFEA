@echo off
title WNFEA Hardware Utilization & Gating Bottleneck Monitor
cd /d "%~dp0"
"C:\Python314\python.exe" gui_hardware_monitor.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] WNFEA Hardware Monitor encountered an error.
    pause
)
