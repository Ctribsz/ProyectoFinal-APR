@echo off
rem Crea .venv en la raiz del repositorio e instala lo minimo para entrenar y afinar.
rem No necesita permisos de administrador. Se puede volver a correr sin problema.
setlocal
cd /d "%~dp0.."
set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    echo Ya existe .venv; se reutiliza.
    goto :instalar
)

rem Busca Python 3.12 (el de la laptop); si no esta, 3.11 o 3.13. 3.14 aun no tiene todas las ruedas.
set "PY="
for %%v in (3.12 3.11 3.13) do (
    if not defined PY (
        py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo No se encontro Python 3.10 a 3.13.
    echo Instalalo desde https://www.python.org/downloads/release/python-31210/
    echo   - Windows installer 64-bit
    echo   - En el instalador: marca "Add python.exe to PATH" y usa "Install Now" ^(no pide administrador^).
    echo Luego vuelve a correr este archivo.
    goto :error
)

echo Creando .venv con: %PY%
%PY% -m venv .venv || goto :error

:instalar
"%VENV_PY%" -m pip install --upgrade pip || goto :error
echo.
echo Instalando torch ^(solo CPU, unos 200 MB^)...
"%VENV_PY%" -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu || goto :error
echo.
echo Instalando el resto...
"%VENV_PY%" -m pip install -r lab\requirements-lab.txt || goto :error

echo.
echo Comprobando que el entorno funciona...
"%VENV_PY%" -c "from envs.balance_bot_env import BalanceBotEnv, load_config; import stable_baselines3, optuna; env = BalanceBotEnv(level=2, config=load_config(2)); env.reset(seed=0); env.step(env.action_space.sample()); print('OK: entorno, SB3', stable_baselines3.__version__, 'y Optuna', optuna.__version__)" || goto :error

echo.
echo Listo. Siguiente paso: medir.bat ^(unos 10 minutos^) y luego afinar.bat.
pause
exit /b 0

:error
echo.
echo Hubo un error. Si fallo la descarga, puede ser el proxy de la red ^(ver lab\README.md^).
pause
exit /b 1
