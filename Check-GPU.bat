@echo off
REM Double-click to run the GPU health check (compute, VRAM, temps).
REM Stresses the GPU for ~60s. Watch GPU Memory Junction temp in HWiNFO64 too.
"%~dp0.venv\Scripts\python.exe" "%~dp0gpu_health_check.py"
echo.
pause
