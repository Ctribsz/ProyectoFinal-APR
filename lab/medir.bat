@echo off
rem Mide los pasos/s de esta computadora (unos 10 minutos). Uso: medir.bat [procesos_en_paralelo]
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Falta .venv: corre primero instalar.bat
    pause
    exit /b 1
)
set PYTHONUTF8=1
if "%~1"=="" (
    ".venv\Scripts\python.exe" -m lab.benchmark
) else (
    ".venv\Scripts\python.exe" -m lab.benchmark --parallel %~1
)
pause
