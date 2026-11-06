@echo off
rem Afina un algoritmo con Optuna usando varios procesos.
rem Uso: afinar.bat [algoritmo] [procesos] [primer_worker]     Sin argumentos, los pregunta.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Falta .venv: corre primero instalar.bat
    pause
    exit /b 1
)

set "ALGO=%~1"
set "WORKERS=%~2"
set "FIRST=%~3"
if "%ALGO%"=="" set /p "ALGO=Algoritmo (ddpg, ppo, td3, sac): "
if "%WORKERS%"=="" set /p "WORKERS=Procesos en paralelo (el que mejor dio medir.bat): "
if "%FIRST%"=="" set "FIRST=0"

set PYTHONUTF8=1
".venv\Scripts\python.exe" -m lab.afinar --algo %ALGO% --workers %WORKERS% --first-worker %FIRST%
echo.
echo Al terminar, copia la carpeta results\optuna a una USB o a OneDrive.
pause
