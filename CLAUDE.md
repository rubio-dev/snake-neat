# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r src/requirements.txt

# Run training
python src/snake_ai/train.py

# Reset all checkpoints and restart from Gen 0
python reset_evolution.py
```

No test suite exists. Validation is visual (GUI) and via stdout fitness logs.

## Architecture

This project trains a Snake-playing neural network using NEAT (NeuroEvolution of Augmenting Topologies).

**Entry point:** `src/snake_ai/train.py`
- Orchestrates two concurrent threads: evolution (background daemon) + GUI (main thread at 60 FPS)
- Loads the latest valid checkpoint from `checkpoints-neat-snake-{config}-v2/` on startup, falling back to prior checkpoints on corruption
- Saves checkpoints every `intervalo_checkpoint` generations and an emergency checkpoint on GUI close

**Hyperparameters:** `src/snake_ai/parametros.py`
- All tunable constants live here; imported by `train.py` and `gui.py`

**Game engine:** `lib/fast_snake/` (submodule)
- Numba-compiled (`@njit`) Snake logic — `generate_game()`, `move_snake()`, `render()`
- Board represented as a numpy array with `TileTypes` enum values

**Perception:** `src/snake_ai/perception.py`
- Builds the input vector for the network using raycasting (Bresenham's line algorithm, Numba-compiled)
- Default layout: `[0-3]` obstacle proximity (per ray), `[4-7]` food flags (per ray), `[8-11]` wall distances (4 sides), `[12-13]` normalized food direction
- Optional extras: last direction, snake length

**Movement:** `src/snake_ai/movement.py`
- `elegir_direccion(salidas, ultima_dir)` — selects best output direction, filtering out 180° reversals

**NEAT config:** `src/snake_ai/config`
- Population: 200 genomes; outputs: 4 (R/L/D/U)
- `num_inputs` must match the value computed by `calcular_tam_entradas()` — the program prints the correct value on startup if there is a mismatch

**GUI reporter:** `src/neat_reporters/gui.py`
- Implements `neat.reporting.BaseReporter`; runs live simulation of best genome each generation
- Fixed 1280×720 window; left panel = game board, right panel = stats + fitness history graph

**Visualization:** `src/neat_reporters/visualization.py`
- `dibujar_red`, `graficar_estadisticas`, `graficar_especies` — used after training completes

## Key Hyperparameters (all in `parametros.py`)

Changing `num_rayos`, any `incluir_*` flag, or `long_temporal` requires updating `num_inputs` in `src/snake_ai/config`.

| Parameter | Default | Effect |
|---|---|---|
| `partidas_por_tam` | 10 | Games per genome per generation |
| `num_rayos` | 4 | Vision rays → changes input size |
| `incluir_dist_pared` | True | +4 inputs |
| `incluir_ultima_dir` | False | +2 inputs |
| `incluir_long_serp` | False | +1 input |
| `long_temporal` | 1 | Frame stacking multiplier on inputs |
| `recompensa_comida` | 500.0 | Points per food eaten |
| `max_pasos_hambre` | board² / 4 | Steps before starvation death |
| `intervalo_checkpoint` | 25 | Generations between saves |
| `habilitar_gui` | True | Disable for headless/faster training |

## Optional System Dependency

Graphviz binary is required for the "Ver Red Neuronal" button (generates `Digraph.svg`). The code auto-detects it in Windows registry paths. Install with: `winget install graphviz`
