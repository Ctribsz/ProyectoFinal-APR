@echo off
rem En la laptop: crea ..\ProyectoFinal-APR-lab.zip con todo lo necesario (sin .venv ni results).
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m lab.empaquetar
pause
