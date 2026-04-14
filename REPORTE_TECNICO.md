# REPORTE TÉCNICO: NEAT SNAKE
## Sistema de Evolución de Redes Neuronales para el Juego Snake

**Fecha:** Abril 2026  
**Proyecto:** NeuroEvolution de Augmenting Topologies aplicado a Snake  
**Lenguaje:** Python 3.11+  
**Framework:** neat-python + Pygame-CE + Numba  

---

## 1. DESCRIPCIÓN GENERAL

### Objetivo
Entrenar redes neuronales con el algoritmo **NEAT** para aprender a jugar Snake de forma autónoma, sin especificar reglas explícitas. La red evoluciona tanto su topología (número de nodos y conexiones) como sus pesos internos.

### Características Principales
- **Evaluación Paralela:** Multiprocessing (configurable 3-6 cores)
- **GUI Interactiva:** Visualización en vivo de simulaciones a 60 FPS
- **Checkpoints Automáticos:** Cada 25 generaciones
- **Recuperación de Fallos:** Fallback automático ante checkpoints corruptos
- **Visualización de Redes:** Exporta topología con Graphviz
- **Estadísticas en Tiempo Real:** Fitness, especies, tiempo/gen

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Estructura de Archivos
```
snake-neat/
├── src/
│   ├── snake_ai/
│   │   ├── ai_multiprocessing.py       # Punto de entrada + loop de NEAT
│   │   ├── inputs.py                   # Generación de observaciones (rayos)
│   │   ├── constants.py                # Direcciones y configuración
│   │   └── config                      # Configuración NEAT (14 inputs, 4 outputs)
│   ├── neat_reporters/
│   │   └── ai_reporter_gui.py          # Renderizado Pygame + simulación live
│   └── requirements.txt
├── lib/
│   └── fast_snake/src/fast_snake/
│       └── fast_snake.py               # Motor de juego (Numba-optimizado)
├── data/
│   └── [archivos de entrenamiento anterior]
├── checkpoints-neat-snake-*/
│   └── [carpetas con checkpoints por generación]
├── config                              # Config NEAT
└── README.md
```

### 2.2 Flujo de Ejecución
```
ai_multiprocessing.py
│
├─ Cargar config NEAT + último checkpoint
├─ Crear population (200 genomas) o restaurar desde checkpoint
├─ Crear ParallelEvaluator (N workers = CPU//4 con GUI, CPU//2 sin GUI)
├─ Lanzar hilo de evolución (background)
│  └─ eval_genome() × 200 × RUNS_PER_GAME_SIZE
│     └─ move_snake() × FOOD_TIMER_MAX pasos
│        └─ GUI renderiza cada frame a 60 FPS
│        └─ Trainer simula en paralelo
├─ GUI loop (60 FPS)
│  ├─ Renderizar best_genome jugando (simulación live)
│  ├─ Mostrar estadísticas
│  │  └─ Generación, Fitness, Tiempo, Especies, Hambre
│  └─ Permitir click "Ver Red Neuronal" → Graphviz
└─ Al cerrar: Guardar checkpoint final
```

---

## 3. EL ALGORITMO NEAT APLICADO A SNAKE

### 3.1 Representación del Genoma

Cada genoma es una red neuronal feedforward definida por:
- **Nodos:** Capas de entrada (14), capas ocultas (variables), salida (4)
- **Arcos:** Conexiones directas con pesos aprendibles
- **Genes:** Mutación estructural (add/delete nodos y conexiones)

**Configuración:**
```
Inputs (14):
├─ Rayos 4 direcciones (distancia a obstáculo) × 4
├─ Flags de comida (en línea del rayo) × 4
├─ Distancia a paredes (x2, x1, x0.5 normalizado) × 4
└─ Vector directo a comida (dx, dy normalizado) × 2

Outputs (4):
├─ Acción 1: Arriba
├─ Acción 2: Abajo
├─ Acción 3: Izquierda
└─ Acción 4: Derecha
(Se ejecuta el argmax de estos 4 valores)

Hidden layers: Creados dinámicamente durante mutación (empiezan en 0)
```

### 3.2 Sistema de Fitness

**Recompensas:**
```python
FOOD_REWARD = 500.0        # Por cada comida consumida
SURVIVAL_BONUS = 0.01      # Por cada paso sin comer (muy pequeño)
FOOD_TIMER_MAX = 80 pasos  # Máximo sin comer (8 comidas × 10 pasos cada)
```

**Métrica Final:**
```
fitness = (total_food × 500 + total_steps × 0.01) / RUNS_PER_GAME_SIZE
```

**Estrategia:** Domina la búsqueda de comida; castiga inmovilidad.

### 3.3 Especiación y Selección

```
Compatibilidad: δ = c₁·D_disj + c₂·D_weight + c₃·D_excess
├─ D_disj: genes desconexión
├─ D_weight: diferencia de pesos
├─ D_excess: genes extras

Threshold: δ > 3.0 → especie diferente
Species Elitism: Protege 1 especie élite (no se extingue)
Stagnation: 30 generaciones sin mejora → extinción automática
```

### 3.4 Evaluación Paralela

```
Generación N:
├─ 200 genomas en población
├─ Dividir en 3-6 workers (según GUI activa)
│  └─ Cada worker evalúa ~33 genomas
│     └─ 1 genoma × 10 partidas (RUNS_PER_GAME_SIZE)
│        └─ 1 partida × 80 pasos máximo
├─ Colectar fitness de todos
├─ Ranking + selección de padres
└─ Reproducción + mutación para Generación N+1
```

---

## 4. COMPONENTES CLAVE

### 4.1 Motor de Juego (`fast_snake.py`)

**Función:** `generate_game(game_size=(40,40))`
- Crea tablero 40×40 con paredes en bordes (38×38 de juego real)
- Inicializa serpiente (1 célula) + comida (1 célula)
- Retorna: `(game_array, food_pos, snake_data)`

**Función:** `move_snake(game_array, snake_data, direction, food_pos)`
```
Entrada: Estado actual del juego, dirección de movimiento
Proceso:
├─ Calcular nueva posición de cabeza
├─ Validar límites del tablero
├─ Checar colisión (pared, cuerpo propio)
├─ Si es comida: crecer, nueva comida
├─ Si está vacío: mover y acortar
└─ Si es obstáculo: devolver dead=True
Salida: (snake_data_updated, is_dead, food_pos, eaten)
```

**Bug Actual:** Off-by-one en actualización de arrays
- Al mover serpiente hacia el tail, hay desincronización entre `game_array` y `snake_data`
- Causaba que la serpiente se viera a sí misma como obstáculo

### 4.2 Generación de Observaciones (`inputs.py`)

**Raycasting (4 direcciones):**
```
De la cabeza de la serpiente, lanzar 4 rayos (arriba, abajo, izq, der)
Para cada rayo:
├─ Distancia invertida al obstáculo (1.0 = pegado, 0.0 = lejos)
└─ Flag booliano: ¿hay comida en este rayo?
```

**Vector Directo:**
```
(dx, dy) normalizado del centro de la serpiente a la comida
```

**Distancias a Paredes (4 inputs):**
```
Distancia normalizada a cada borde del tablero
```

**Total:** 4×2 + 2 + 4 = 14 inputs

### 4.3 GUI (`ai_reporter_gui.py`)

**Arquitectura:**
```
Pygame Window (1280×720)
├─ Panel izquierdo (40%): Tablero de juego
│  ├─ Renderiza game_array a escala visual
│  ├─ Simula best_genome a 25 pasos/seg (0.04s/paso)
│  └─ Mantiene genome congelado durante partida visual
├─ Panel derecho (60%): Estadísticas
│  ├─ Generación actual
│  ├─ Mejor fitness
│  ├─ Tiempo por generación
│  ├─ Número de especies
│  ├─ Barra de hambre (FOOD_TIMER)
│  └─ Botón: Ver Red Neuronal
└─ Loop: 60 FPS render, ~25 FPS juego (desacoplados)
```

**Sincronización GUI↔Training:**
- Train thread: eval_genome() sin GUI
- GUI thread: visualiza best_genome de forma estable
- Cada generación en training → reinicia partida visual
- Cada muerte visual → reinicia desde mismo genoma

---

## 5. BUGS DETECTADOS Y DIAGNOSTICADOS

### 5.1 Bug Principal: Desincronización en `move_snake()`

**Síntoma:** La serpiente muere sin colisionar visualmente

**Causa:** Al actualizar `game_array` y `snake_data` en orden incorrecto
```python
# INCORRECTO (actual):
game_array[new_head_pos] = SNAKE_HEAD
game_array[old_head_pos] = SNAKE_BODY
game_array[old_tail_pos] = EMPTY      # Puede borrar la nueva cabeza si tail==new_head
snake_data = agregar_cabeza + remover_tail

# CORRECTO (esperado):
snake_data = agregar_cabeza + remover_tail  # Primero actualizar lógica
game_array[new_tail_pos] = EMPTY           # Borrar el tail calculado correctamente
game_array[new_head_pos] = SNAKE_HEAD      # Marcar nueva cabeza
```

**Impacto:** Cuando la IA intenta moverse hacia el tail (comienzo del cuerpo), 
ambos arrays se desincronizaban, creando un estado inválido.

### 5.2 Bug Secundario: Victoria Incompleta

**Síntoma:** No se detecta victoria cuando el tablero está lleno

**Resultado de `get_rand_empty_pos()`:**
```
Si no hay celdas vacías → devuelve (-1, -1)
Esto indica: "no hay dónde poner comida" = tablero lleno = victoria
```

**Status:** Detectado, esperando ajuste en `move_snake()`

### 5.3 Problema de Diseño: Partida Visual Termina Rápido

**Síntoma:** Generación cambia muy rápido en la GUI (segundos)

**Causa:** 
- Training thread aprox. 30-60 segundos por generación (100 CPU)
- GUI visualiza solo 1 partida por generación (80 pasos = 3.2 segundos a 25 FPS)
- Desacoplamiento intencional: GUI muestra "muestras" del progreso

**Intención Original:** Demostrar progreso visual sin impactar performance

**Problema Reportado:** Usuario espera ver juegos completos/largos

---

## 6. MODIFICACIONES REALIZADAS

### 6.1 Correcciones Aplicadas

1. **Sincronización de Checkpoints (línea ~340)**
   ```python
   # Incrementar generación solo DESPUÉS de guardar checkpoint
   # (evita perder la generación actual si falla carga)
   ```

2. **Reducción de CPU para GUI (línea ~120)**
   ```python
   if ENABLE_GUI:
       CORE_COUNT = max(1, multiprocessing.cpu_count() // 4)
   else:
       CORE_COUNT = max(1, multiprocessing.cpu_count() // 2)
   ```
   Justificación: Prioriza render thread

3. **Lazy Imports (línea ~305)**
   ```python
   # import pygame movido dentro de render() 
   # import ai_reporter_gui solo si ENABLE_GUI=True
   ```
   Justificación: Reduce overhead inicial

4. **Eliminación de Desviación Estándar en GUI**
   ```python
   # Removido cálculo de fitness_std en post_evaluate()
   # Removida visualización de "Desviación Estándar"
   ```

---

## 7. PROBLEMAS PENDIENTES

### P1: Serpiente Muere Sin Razón (CRÍTICO)
**Estado:** Parcialmente arreglado  
**Último Intento:** Reordenamiento de operaciones en `move_snake()`  
**Evidencia:** Debug output mostraba "Choque con cuerpo" cuando new_head==tail  
**Solución Propuesta:** Ver sección 5.1  

### P2: Partida Visual Termina Muy Rápido
**Estado:** Por diseño, pero confunde al usuario  
**Solución Propuesta:** 
- Aumentar `FOOD_TIMER_MAX` de 80 a 625 (~25 segundos a 25 FPS)
- O: Congelar pantalla 2-3 segundos cuando la serpiente muere

### P3: NEAT Avanza Generaciones Sin Evaluación Visible
**Estado:** Comportamiento normal pero no intuitivo  
**Causa:** NEAT train thread corre en background sin sincronización visual  
**Solución:** Mostrar "Gen N/10 Evaluandose..." en GUI  

---

## 8. PARÁMETROS DE CONFIGURACIÓN

### Entrenamiento
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `pop_size` | 200 | Población mediana, balance entre diversidad y velocidad |
| `RUNS_PER_GAME_SIZE` | 10 | 10 partidas/genoma para robustez |
| `FOOD_TIMER_MAX` | 80 | 80 pasos ≈ 3.2 seg simulados |
| `CHECKPOINT_INTERVAL` | 25 | Cada 25 generaciones (balance I/O) |
| `max_stagnation` | 30 | Antes era 20, aumentado para estrategias complejas |

### Red Neuronal
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `weight_mutate_rate` | 0.8 | Alta para exploración agresiva |
| `conn_add_prob` | 0.2 | Permite crecer redes complejas |
| `compatibility_threshold` | 3.0 | Especiación moderada |
| `activation_default` | tanh | Saturación suave vs relu agresivo |

### GUI
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `fps_limit` | 60 | Render smooth |
| `sim_step_seconds` | 0.04 | 25 FPS de simulación |
| `max_steps_per_frame` | 2 | Limita backlog sin bloquear render |

---

## 9. RECOMENDACIONES PARA INVESTIGADORES

### Hipótesis de Problemas
1. **Desincronización Arrays:** Validar que `game_array[pos] == snake_data` siempre
2. **Collision Detection:** Checar que HEAD no se sobrescriba con TAIL en mismo paso
3. **Temporal Coherence:** GPU threads pueden sobreescribir simultáneamente

### Tests Propuestos
```python
# Debug: Dump game_array y snake_data después de cada move
# Checar invariantes:
# - snake_data[0] debe estar marcado como BODY en game_array
# - snake_data[-1] debe estar marcado como HEAD
# - No debe haber SNAKE_HEAD múltiples
# - Longitud de snake_data debe coincidir con COUNT(BODY) + 1
```

### Arquitectura Alternativa
1. Usar una clase `GameState` inmutable (retorna nuevo estado, no muta)
2. Separar "lógica de juego" (move_snake) de "renderizado" (game_array)
3. Usar `dataclass` con `frozen=True` para evitar mutaciones accidentales

### Mejoras Propuestas
1. Aumentar `FOOD_TIMER_MAX` para permitir estrategias de exploración
2. Implementar "reward shaping": bonus por distancia a comida
3. Multi-objetivo: maximizar comida Y distancia explorada
4. Usar redes convolucionales 1D en lugar de fully connected

---

## 10. COMANDOS Y ARCHIVOS DE REFERENCIA

### Ejecutar Entrenamiento
```bash
cd c:\Users\rubio\Documents\Escuela\snake-neat
.\venv\Scripts\python.exe src\snake_ai\ai_multiprocessing.py
```

### Ver Checkpoint Actual
```bash
ls checkpoints-neat-snake-4-rays-False-False-True-False-1-v2/
```

### Resetear Entrenamiento
```bash
python reset_evolution.py
```

### Archivos Críticos
- `src/snake_ai/ai_multiprocessing.py` - Main loop NEAT (500+ líneas)
- `src/neat_reporters/ai_reporter_gui.py` - GUI Pygame (800+ líneas)
- `lib/fast_snake/src/fast_snake/fast_snake.py` - Motor juego (150 líneas, crítico)
- `src/snake_ai/constants.py` - Direcciones y configuración
- `src/snake_ai/inputs.py` - Raycasting y observaciones (200+ líneas)

---

## 11. CONTACTO Y NOTAS

**Última Actualización:** Abril 13, 2026  
**Status del Proyecto:** En debugging  
**Bloqueador Principal:** Desincronización move_snake() ← CRÍTICO  

Para consultar: Contactar a investigador original del proyecto.

---

## APÉNDICE: Código Problemático

### move_snake() - Sección Problemática (Líneas 107-115)
```python
# PROBLEMA: Orden de operaciones
if game_array[new_head_pos] == TileTypes.EMPTY.value:
    game_array[new_head_pos] = TileTypes.SNAKE_HEAD.value      # ← Marca nueva cabeza
    game_array[head_pos[0], head_pos[1]] = TileTypes.SNAKE_BODY.value  # ← Antigüa cabeza→cuerpo
    # AQUÍ: Si tail_pos == new_head_pos (intenta moverse hacia atrás)
    #       La siguiente línea borra la SNAKE_HEAD que acabamos de poner
    game_array[tail_pos[0], tail_pos[1]] = TileTypes.EMPTY.value  # ← BORRA NUEVA CABEZA
    
    snake_data = np.concatenate([snake_data, [new_head_pos]], axis=0)
    snake_data = snake_data[1:]
    # Resultado: game_array corrupto, snake_data incorrecto
```

### Solución Propuesta (Reorder)
```python
# CORRECTO: Actualizar lógica primero, luego render
old_tail_pos = snake_data[0].copy()
snake_data = np.concatenate([snake_data, [new_head_pos]], axis=0)  # Agregar cabeza NEW
snake_data = snake_data[1:]                                         # Remover tail OLD

# Ahora el nuevo tail es snake_data[0], viejo tail es old_tail_pos
# Marcar el viejo tail como EMPTY (ya no está en snake_data)
game_array[old_tail_pos[0], old_tail_pos[1]] = TileTypes.EMPTY.value

# Marcar nueva cabeza
game_array[new_head_pos] = TileTypes.SNAKE_HEAD.value

# Marcar antigua cabeza como cuerpo
game_array[head_pos[0], head_pos[1]] = TileTypes.SNAKE_BODY.value
```

---

*Fin del Reporte Técnico*
