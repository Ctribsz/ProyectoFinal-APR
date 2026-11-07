# Correr el afinamiento en otra computadora (laboratorio)

Herramientas para correr `training.tune` en computadoras de la universidad u otras máquinas con Windows. No
necesitan permisos de administrador, no usan PowerShell y no descargan ejecutables: solo Python oficial y
paquetes de PyPI.

## 1. En la laptop: preparar el paquete

Doble clic en `lab\empaquetar.bat`. Crea `ProyectoFinal-APR-lab.zip` junto a la carpeta del repo (≈0.4 MB,
sin `.venv` ni `results/`). Llévalo en una USB, por correo o por OneDrive.

> También sirve descargar el repo de GitHub con *Code → Download ZIP*, si los cambios ya están subidos.

## 2. En la computadora del laboratorio

1. **Desbloquear el .zip antes de descomprimir:** clic derecho → *Propiedades* → marcar *Desbloquear* →
   *Aceptar*. Así Windows no muestra "Windows protegió su PC" en cada `.bat`.
2. Descomprimir en una carpeta **local** (por ejemplo `C:\Users\<usuario>\Documents`), no en una unidad de
   red: SQLite no funciona bien en carpetas compartidas.
3. **`lab\instalar.bat`** (doble clic). Busca Python 3.12 (o 3.11/3.13), crea `.venv` e instala todo
   (unos 5 minutos y ~1 GB). Si no hay Python, el mensaje dice de dónde bajarlo: el instalador de python.org
   con *Install Now* no pide administrador.
4. **`lab\medir.bat`** (unos 10 minutos, sin usar la PC). Muestra los pasos/s de DDPG y PPO y el
   rendimiento con varios procesos a la vez. Para probar otro número de procesos: `medir.bat 6` desde una
   ventana de cmd en la carpeta `lab`.
5. **`lab\afinar.bat`** (doble clic). Pregunta el algoritmo y cuántos procesos usar; reparte las 40 pruebas
   de `configs\training\tune.yaml` entre ellos. Cada minuto muestra cuántas pruebas van y la mejor.

## 3. Al terminar: traer los resultados

Copia la carpeta `results\optuna` completa (el `.db`, una carpeta por prueba y `logs\`) a la misma ruta del
repo en la laptop. Con eso se analiza el estudio y se elige la mejor configuración (E10).

## Recomendaciones

- **Una computadora por algoritmo** (por ejemplo, DDPG en una y PPO en otra). Varios procesos en la misma
  PC comparten el `.db` sin problema; varias PC no pueden compartir uno por red.
- **Procesos:** el que dé el mayor "pasos/s en total" en `medir.bat`. Suele ser cerca del número de núcleos
  físicos.
- **No cerrar sesión ni apagar.** `afinar` evita que la PC se suspenda mientras corre, y bloquear la pantalla
  no detiene nada. Cerrar sesión, reiniciar o apagar sí lo detiene.
- **Si se interrumpe** (por ejemplo, la PC se reinició), se vuelve a lanzar igual (`afinar.bat`). El estudio
  continúa desde el `.db`: las pruebas que quedaron a medias pasan a FAIL y no cuentan, se lanzan solo las que
  faltan para llegar a `n_trials` y los workers usan números nuevos, para que no repitan semillas. Si dice que
  ya hay un afinamiento corriendo y no es cierto, espera 2 minutos y vuelve a lanzarlo.
- Algunas PC de laboratorio borran todo al reiniciar (Deep Freeze). Copia `results\optuna` a la USB
  seguido, y antes de irte.
- Confirma con el laboratorio que se pueden dejar procesos corriendo varias horas.

## Problemas comunes

| Síntoma | Qué hacer |
|---|---|
| `pip` no puede descargar (timeout, SSL) | La red usa proxy. En cmd: `set HTTPS_PROXY=http://<proxy>:<puerto>` y volver a correr `instalar.bat` desde esa misma ventana |
| "Windows protegió su PC" | *Más información → Ejecutar de todas formas*, o desbloquear el .zip (paso 2.1) |
| El antivirus pone en cuarentena algo de `.venv` | Pasa a veces con DLL de torch. Reportarlo como falso positivo o probar en otra carpeta |
| Un worker terminó con error | Revisar `results\optuna\logs\<algo>_worker<i>.log` |
