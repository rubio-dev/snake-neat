# NEAT Snake

Sistema de entrenamiento de Inteligencia Artificial para el juego Snake utilizando
el algoritmo **NEAT** (NeuroEvolution of Augmenting Topologies).

## Descripción

Este proyecto evoluciona redes neuronales para que un agente aprenda a jugar Snake
de forma autónoma. Utiliza **procesamiento paralelo** (multiprocessing) para evaluar
genomas en todos los núcleos disponibles y cuenta con una **interfaz gráfica
interactiva** que permite visualizar el progreso en tiempo real.

## Características Principales

- Entrenamiento continuo con evaluación paralela multi-núcleo.
- Interfaz gráfica (GUI) basada en Pygame-CE y Pygame GUI con ventana fija de 1280×720.
- Sistema de checkpoints automáticos cada 5 generaciones para reanudar sin perder progreso.
- Recuperación automática ante checkpoints corruptos (fallback al anterior).
- Visualización de la topología de la red neuronal (requiere Graphviz instalado).
- Estadísticas en tiempo real: fitness, generación, tiempo/gen, especies, barra de hambre.
- Simulación en vivo del mejor genoma de la generación actual.

## Requisitos del Sistema

- Python 3.11+
- Graphviz instalado como herramienta de sistema (y en el PATH) para visualizar redes.
- Dependencias de Python (ver `src/requirements.txt`).

## Instalación

```powershell
# 1. Crear entorno virtual
python -m venv venv

# 2. Activarlo
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias de Python
pip install -r src/requirements.txt
```

## Ejecución

```powershell
python src/snake_ai/ai_multiprocessing.py
```

El programa carga automáticamente el último checkpoint válido. Si no existe ninguno,
inicia la evolución desde la Generación 0.

Para **reiniciar** la evolución desde cero:

```powershell
python reset_evolution.py
```

## Controles de la GUI

| Acción | Descripción |
|---|---|
| Cerrar ventana | Detiene el entrenamiento de forma segura y guarda checkpoint final |
| Botón **Ver Red Neuronal** | Genera y abre el diagrama SVG de la red del mejor genoma |

## Algoritmo de Fitness

Cada genoma juega **10 partidas** en un tablero de 40×40 y su fitness es el promedio:

| Evento | Recompensa |
|---|---|
| Comer una pieza de comida | +500.0 puntos |
| Sobrevivir un paso sin comer | +0.01 puntos |
| Timeout sin comer (> `FOOD_TIMER_MAX` pasos) | Muerte por hambre |
| Victoria (tablero lleno) | +500.0 y fin inmediato |

## Parámetros Configurables

Editar en `src/snake_ai/ai_multiprocessing.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `RUNS_PER_GAME_SIZE` | `10` | Partidas por genoma por generación |
| `ENABLE_GUI` | `True` | Activar/desactivar la GUI |
| `NUMBER_OF_RAYS` | `4` | Rayos de visión de la serpiente |
| `INCLUDE_WALL_DISTANCE` | `True` | Incluir distancias a paredes como input |
| `INCLUDE_LAST_DIRECTION` | `False` | Incluir última dirección como input |
| `INCLUDE_SNAKE_LENGTH` | `False` | Incluir longitud normalizada como input |
| `TEMPORAL_LENGTH` | `1` | Frames concatenados (1 = sin memoria temporal) |
| `FOOD_REWARD` | `500.0` | Recompensa por comer |
| `FOOD_TIMER_MAX` | `grid_size² / 4` (400 para 40×40) | Pasos máximos sin comer (timeout de hambre); calculado dinámicamente |
| `CORE_COUNT` | `CPU // 2` | Núcleos para evaluación paralela |

> **Nota:** Si cambias `NUMBER_OF_RAYS`, `INCLUDE_*` o `TEMPORAL_LENGTH`,
> debes actualizar `num_inputs` en `src/snake_ai/config`. El programa valida
> esto al arrancar y reporta el valor correcto si hay discrepancia.

## Arquitectura del Proyecto

```
snake-neat/
├── src/
│   ├── snake_ai/
│   │   ├── ai_multiprocessing.py   # Punto de entrada y orquestador principal
│   │   ├── inputs.py               # Generación del vector de observación (raycasting)
│   │   ├── constants.py            # Direcciones y constantes compartidas
│   │   └── config                  # Configuración NEAT (parámetros de evolución)
│   └── neat_reporters/
│       ├── ai_reporter_gui.py      # Reporter + GUI Pygame en tiempo real
│       ├── visu.py                 # Visualización de redes y estadísticas
│       └── utils.py                # Utilidades de imagen auxiliares
├── lib/fast_snake/                 # Motor de juego Snake (submódulo C/Cython)
├── data/                           # Assets: fuentes, sonidos, imágenes
├── reset_evolution.py              # Script para borrar checkpoints y reiniciar
└── checkpoints-neat-snake-*/       # Carpetas de checkpoints (generadas al entrenar)
```

## Requisitos opcionales

- **Graphviz** (herramienta de sistema): necesario para el botón "Ver Red Neuronal".
  - Windows: `winget install graphviz`
  - El programa detecta automáticamente las rutas de instalación típicas.
