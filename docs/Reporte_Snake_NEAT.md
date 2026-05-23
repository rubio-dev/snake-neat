# Snake NEAT AI — Reporte de Proyecto

**Entrenamiento de una Inteligencia Artificial para jugar Snake usando Neuro-Evolución**

| Campo | Detalle |
|---|---|
| **Fecha** | Mayo 2026 |
| **Lenguaje** | Python 3.11+ |
| **Algoritmo principal** | NEAT (NeuroEvolution of Augmenting Topologies) |
| **Librerías clave** | neat-python, Numba, NumPy, Pygame-CE, Graphviz, Matplotlib |
| **Estado** | En entrenamiento activo (~6,300+ generaciones) |

---

## Índice

1. [Introducción](#1-introducción)
2. [¿Qué es NEAT?](#2-qué-es-neat)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Percepción: cómo "ve" la serpiente](#4-percepción-cómo-ve-la-serpiente)
5. [El cerebro: red neuronal](#5-el-cerebro-red-neuronal)
6. [Sistema de movimiento con respaldo](#6-sistema-de-movimiento-con-respaldo)
7. [Función de fitness: cómo aprende](#7-función-de-fitness-cómo-aprende)
8. [Interfaz gráfica](#8-interfaz-gráfica)
9. [Evaluación paralela y rendimiento](#9-evaluación-paralela-y-rendimiento)
10. [Checkpoints y continuidad](#10-checkpoints-y-continuidad)
11. [Stack tecnológico y dependencias](#11-stack-tecnológico-y-dependencias)
12. [Guía de ejecución](#12-guía-de-ejecución)
13. [Evolución del diseño (v1 → v2)](#13-evolución-del-diseño-v1--v2)
14. [Problemas conocidos y bugs](#14-problemas-conocidos-y-bugs)
15. [Trabajo futuro](#15-trabajo-futuro)

---

## 1. Introducción

### 1.1 ¿De qué trata el proyecto?

Este proyecto entrena una **inteligencia artificial para jugar al videojuego clásico Snake** de forma completamente autónoma. La IA no recibe reglas preprogramadas sobre cómo moverse, evitar paredes, perseguir comida o esquivar su propio cuerpo. En lugar de eso, **descubre todas esas estrategias por sí misma** a través de un proceso de evolución simulada basado en selección natural.

### 1.2 El problema

Snake es un juego de control en tiempo real con un espacio de estado enorme. En un tablero de 40×40 hay aproximadamente **1,600 celdas**, y la serpiente puede ocupar cualquier subconjunto de ellas en cualquier orden. El número de configuraciones posibles es astronómico, lo que hace impracticable cualquier enfoque de tablas de valores o búsqueda exhaustiva.

Además, Snake presenta un problema de **señal escasa**: la recompensa (comer comida) ocurre con poca frecuencia, especialmente al inicio del entrenamiento. La IA necesita un mecanismo que le permita aprender incluso cuando los eventos de retroalimentación son raros.

### 1.3 La solución: neuro-evolución

Para resolver este problema utilizamos **NEAT (NeuroEvolution of Augmenting Topologies)**, un algoritmo que combina **redes neuronales artificiales** con **algoritmos genéticos**. En lugar de entrenar una red mediante retropropagación (que requiere un conjunto de datos etiquetados), NEAT **evoluciona poblaciones de redes neuronales** usando principios biológicos:

- **Selección natural**: las redes que mejor juegan sobreviven y se reproducen
- **Mutación**: los pesos, conexiones y neuronas cambian aleatoriamente
- **Cruce**: dos redes parentales se combinan para crear descendencia
- **Especiación**: redes similares se agrupan para proteger innovaciones

La única retroalimentación que recibe la IA es una **puntuación de fitness** al final de cada partida, que mide qué tan bien jugó.

### 1.4 Objetivos del proyecto

| Objetivo | Descripción |
|---|---|
| **Principal** | Demostrar la aplicabilidad de NEAT para problemas de control en tiempo real con señal escasa |
| **Secundario** | Crear una herramienta visual interactiva que muestre el progreso del aprendizaje en vivo |
| **Técnico** | Implementar un sistema modular con motor de juego optimizado (Numba) y percepción por raycasting |

---

## 2. ¿Qué es NEAT?

### 2.1 Definición

NEAT (NeuroEvolution of Augmenting Topologies) fue propuesto por **Kenneth Stanley y Risto Miikkulainen en 2002**. Es un algoritmo de neuro-evolución que **evoluciona simultáneamente los pesos y la topología** (arquitectura) de redes neuronales.

A diferencia del Deep Learning tradicional:

| Aspecto | Deep Learning (Backpropagation) | NEAT (Evolución) |
|---|---|---|
| Arquitectura | Fija, definida antes del entrenamiento | Crece y se adapta durante la evolución |
| Retroalimentación | Necesita gradientes y etiquetas | Solo necesita una puntuación (fitness) |
| Datos | Requiere millones de ejemplos etiquetados | Aprende por prueba y error |
| Optimización | Minimiza una función de error | Maximiza una función de fitness |
| Exploración | Limitada a la arquitectura inicial | Puede descubrir arquitecturas novedosas |

### 2.2 El ciclo evolutivo

Cada generación sigue este proceso:

```
POBLACIÓN INICIAL (200 genomas aleatorios)
         │
         ▼
CADA GENOMA JUEGA 10 PARTIDAS DE SNAKE
         │
         ▼
CÁLCULO DE FITNESS (puntuación promedio)
         │
         ▼
ESPECIACIÓN (agrupar genomas similares)
         │
         ▼
SELECCIÓN Y REPRODUCCIÓN
   ├── Los mejores de cada especie pasan a la siguiente generación
   ├── Cruce: combina conexiones de dos padres
   └── Mutación: altera pesos, agrega o elimina conexiones/neuronas
         │
         ▼
NUEVA POBLACIÓN (200 genomas) → GENERACIÓN N+1
```

### 2.3 Especiación

Cuando un genoma muta y adquiere una conexión o neurona nueva, al principio su rendimiento suele ser **peor** que el de sus padres (porque aún no ha aprendido a usar bien la nueva estructura). Sin especiación, ese genoma innovador sería eliminado inmediatamente.

NEAT soluciona esto agrupando genomas en **especies** según su similitud genética. Dentro de cada especie, los genomas compiten solo entre sí, lo que da tiempo a las innovaciones para madurar.

La distancia de compatibilidad entre dos genomas se calcula como:

```
δ = c₁·(E/N) + c₂·(D/N) + c₃·W̄

Donde:
  E = genes en exceso (conexiones que están en un genoma pero no en el otro,
      más allá del índice del genoma más grande)
  D = genes disjuntos (conexiones que caen dentro del rango del genoma más
      grande pero no están alineadas)
  N = número de genes del genoma más grande (normalización)
  W̄ = diferencia promedio de pesos en genes compartidos
  c₁ = c₂ = 1.0,  c₃ = 0.6
```

Si `δ > 3.0`, los genomas pertenecen a especies diferentes. Cada especie que no mejora su fitness durante 30 generaciones se extingue automáticamente.

### 2.4 ¿Por qué NEAT para Snake?

Snake no tiene un conjunto de entrenamiento con respuestas correctas. No podemos decirle a la IA "aquí deberías haber girado a la izquierda". Solo podemos evaluar el resultado final de una partida completa. Esto hace que Snake sea un problema ideal para **aprendizaje por refuerzo evolutivo**, donde NEAT brilla porque:

1. No necesita definir una función de valor estado-acción
2. No sufre de inestabilidad en el entrenamiento (como DQN)
3. Puede descubrir arquitecturas de red adaptadas al problema
4. Es tolerante a recompensas con ruido y esporádicas

---

## 3. Arquitectura del sistema

### 3.1 Estructura de archivos

```
snake-neat/
├── src/
│   ├── snake_ai/
│   │   ├── train.py                    # Punto de entrada: orquesta evolución + GUI
│   │   ├── perception.py               # Raycasting y vector de observación
│   │   ├── movement.py                 # Selector de dirección con 5 niveles de fallback
│   │   ├── parametros.py               # Hiperparámetros globales de entrenamiento
│   │   ├── config                      # Configuración del algoritmo NEAT (texto plano)
│   │   └── __init__.py
│   ├── neat_reporters/
│   │   ├── gui.py                      # Reporter NEAT con GUI Pygame en tiempo real
│   │   ├── ai_reporter_gui.py          # Alias de compatibilidad
│   │   ├── visualization.py            # Visualización de red (Graphviz) y estadísticas
│   │   └── __init__.py
│   └── requirements.txt
├── lib/
│   └── fast_snake/
│       └── src/fast_snake/
│           ├── fast_snake.py           # Motor de juego compilado con Numba JIT
│           └── __init__.py
├── data/
│   ├── gui_theme.json                  # Tema visual de la GUI
│   └── fonts/                          # Tipografías para la interfaz
├── docs/                               # Documentación y reportes
├── checkpoints-neat-snake-8-rays-.../  # Checkpoints de entrenamiento por configuración
├── reset_evolution.py                  # Script para reiniciar el entrenamiento
├── README.md
└── .gitignore
```

### 3.2 Flujo de ejecución

```
python src/snake_ai/train.py
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Calcular num_inputs según la configuración de parámetros │
│    8 rayos × 3 (cuerpo + pared + comida) = 24              │
│    + 4 distancias a paredes                                 │
│    + 2 vector dirección a comida                            │
│    = 30 inputs totales                                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Validar contra el archivo config                         │
│    Si hay mismatch → error claro y termina                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Buscar checkpoint más reciente                           │
│    ├── Si existe → restaurar población                      │
│    ├── Si falla → probar el checkpoint anterior (fallback)  │
│    └── Si no hay → nueva población de 200 genomas           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Lanzar HILO DE EVOLUCIÓN (daemon, background)            │
│                                                             │
│   Por cada generación:                                      │
│   ├── Evaluar 200 genomas × 10 partidas en PARALELO         │
│   │   (usando multiprocessing con CPU//4 núcleos)           │
│   ├── Para cada partida:                                    │
│   │   ├── Generar tablero y serpiente                        │
│   │   ├── Por cada paso:                                    │
│   │   │   ├── Obtener percepción (30 valores)               │
│   │   │   ├── Activar red neuronal                          │
│   │   │   ├── Elegir dirección con sistema de seguridad     │
│   │   │   ├── Ejecutar movimiento                           │
│   │   │   └── Acumular recompensa                           │
│   │   └── Hasta morir, hambre o victoria                    │
│   ├── Calcular fitness promedio                             │
│   ├── NEAT: selección, cruce, mutación, especiación         │
│   ├── Guardar checkpoint cada 25 generaciones               │
│   └── Registrar métricas en CSV                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. HILO PRINCIPAL (GUI a 60 FPS)                           │
│                                                             │
│   ├── Renderizar tablero con el mejor genoma jugando        │
│   ├── Mostrar estadísticas en panel derecho                 │
│   ├── Actualizar cada vez que termina una generación        │
│   └── Al cerrar ventana: guardar checkpoint final           │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Diagrama de componentes

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  perception  │────►│    movement      │◄────│   train.py       │
│  .py         │     │  .py             │     │  (orquestador)   │
│              │     │                  │     │                  │
│ Raycasting   │     │ NEAT → Zigzag →  │     │ Carga NEAT       │
│ Bresenham    │     │ Flood Fill → Safe│     │ ParallelEvaluator│
│ Numba @njit  │     │ → Libre          │     │ Hilo evolución   │
└──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘
       │                      │                        │
       ▼                      ▼                        ▼
┌──────────────────────────────────────────────────────────────┐
│                    fast_snake.py                              │
│               Motor de juego (Numba @njit)                    │
│  generate_game() → move_snake() → render()                   │
└──────────────────────────────────────────────────────────────┘
                           ▲
                           │
┌──────────────────────────────────────────────────────────────┐
│                    neat_reporters/gui.py                      │
│               GUI Pygame (hilo principal, 60 FPS)             │
│  Panel izquierdo: tablero del mejor genoma                    │
│  Panel derecho: estadísticas, gráficas, estado en vivo        │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Percepción: cómo "ve" la serpiente

### 4.1 Principio general

La serpiente **no ve el tablero completo**. En su lugar, utiliza un sistema de **raycasting egocéntrico**: lanza 8 rayos desde su cabeza en 360 grados, y cada rayo detecta qué hay en esa dirección. Es equivalente a tener un radar o un sensor de proximidad.

### 4.2 Raycasting con Bresenham

Cada rayo se traza usando el **algoritmo de Bresenham**, que recorre el tablero celda por celda desde la cabeza hacia afuera en línea recta. Este algoritmo está **compilado con Numba (`@njit`)** para ejecutarse a velocidad nativa.

```
Algoritmo de Bresenham:

Dados (x₁, y₁) y (x₂, y₂):
  dx = |x₂ - x₁|
  dy = |y₂ - y₁|
  sx = 1 si x₁ < x₂, -1 en otro caso
  sy = 1 si y₁ < y₂, -1 en otro caso
  error = dx / 2

Repetir hasta alcanzar (x₂, y₂):
  Registrar (x, y) como punto del rayo
  error -= dy
  Si error < 0:
    y += sy
    error += dx
  x += sx
```

Cada rayo se extiende **2 veces el tamaño del tablero** para garantizar que atraviese cualquier trayectoria.

### 4.3 Información por rayo

Por cada uno de los 8 rayos, la red recibe **tres valores independientes**:

| Valor | Rango | Descripción |
|---|---|---|
| `dist_cuerpo[i]` | [0.0, 1.0] | Proximidad al **cuerpo propio** en esa dirección |
| `dist_pared[i]` | [0.0, 1.0] | Proximidad a la **pared** en esa dirección |
| `band_comida[i]` | {0.0, 1.0} | Flag: ¿hay **comida** en esa dirección? |

**Separar cuerpo y pared es una decisión de diseño importante**: la pared es estática y siempre letal; el cuerpo cambia de forma cada paso y representa un peligro más dinámico. La red puede modelar estas dos amenazas por separado.

### 4.4 Normalización de distancias

La proximidad se calcula con una **transformación hiperbólica**:

```
proximidad = clip(1 / max(distancia_euclidiana, 1.0), 0.0, 1.0)
```

Esto produce una señal **muy sensible cuando el obstáculo está cerca** (1, 2, 3 celdas) y **casi nula cuando está lejos**. La red aprende a ignorar peligros remotos y concentrarse en los inminentes.

### 4.5 Rayos egocéntricos (rotativos)

Con `dir_rotativas = True` (configuración actual), **todos los rayos rotan con la dirección de movimiento** de la serpiente:

```
angulo_inicial = atan2(ultima_dir.fila, ultima_dir.columna)
```

El rayo 0 siempre apunta **hacia adelante** (en la dirección del movimiento). Esto convierte la percepción de coordenadas absolutas a coordenadas **relativas**, permitiendo que la red aprenda patrones como "hay peligro adelante" en lugar de "hay peligro al norte", lo que **generaliza mejor** entre diferentes orientaciones.

### 4.6 Información adicional

Además de los 8 rayos, la red recibe:

| Componente | # Valores | Rango | Descripción |
|---|---|---|---|
| Distancia a pared izquierda | 1 | [0.0, 1.0] | `cabeza_x / ancho_tablero` |
| Distancia a pared derecha | 1 | [0.0, 1.0] | `(ancho - 1 - cabeza_x) / ancho` |
| Distancia a pared superior | 1 | [0.0, 1.0] | `cabeza_y / alto_tablero` |
| Distancia a pared inferior | 1 | [0.0, 1.0] | `(alto - 1 - cabeza_y) / alto` |
| Vector dx hacia comida | 1 | [-1.0, 1.0] | Componente X normalizada |
| Vector dy hacia comida | 1 | [-1.0, 1.0] | Componente Y normalizada |

### 4.7 Vector de entrada completo (30 valores)

```
Índices    Componente                       Valores
────────   ──────────────────────────────   ───────────
[0-7]      Proximidad a CUERPO (8 rayos)    [0.0, 1.0]
[8-15]     Proximidad a PARED (8 rayos)     [0.0, 1.0]
[16-23]    Flag de COMIDA (8 rayos)         {0.0, 1.0}
[24-27]    Distancia a 4 paredes            [0.0, 1.0]
[28-29]    Vector dirección a comida        [-1.0, 1.0]
```

**Fórmula**: `num_inputs = num_rayos × 3 + 4 + 2 = 8 × 3 + 4 + 2 = 30`

---

## 5. El cerebro: red neuronal

### 5.1 Estructura inicial

Cada genoma comienza como una red neuronal **feedforward** (acíclica) con:

| Componente | Cantidad |
|---|---|
| Neuronas de entrada | 30 (vector de percepción completo) |
| Neuronas ocultas | 0 (NEAT las agrega mediante mutación) |
| Neuronas de salida | 4 (Derecha, Izquierda, Abajo, Arriba) |
| Conexiones iniciales | 50% de las posibles (seleccionadas al azar) |
| Activación por defecto | `tanh` |
| Activaciones posibles | `tanh`, `relu` (puede mutar) |

### 5.2 Configuración NEAT

| Parámetro | Valor | Efecto |
|---|---|---|
| `pop_size` | 200 | Tamaño de población por generación |
| `num_inputs` | 30 | Tamaño del vector de percepción |
| `num_hidden` | 0 | Sin ocultas al inicio (NEAT las agrega) |
| `num_outputs` | 4 | Derecha, Izquierda, Abajo, Arriba |
| `initial_connection` | `partial 0.5` | 50% de conexiones iniciales |
| `feed_forward` | `True` | Red acíclica |
| `activation_default` | `tanh` | Función de activación base |
| `activation_mutate_rate` | 0.05 | 5% de mutar la activación |
| `weight_mutate_rate` | 0.8 | 80% de mutar un peso existente |
| `weight_replace_rate` | 0.1 | 10% de reemplazar completamente el peso |
| `weight_init_stdev` | 1.0 | Desviación estándar de pesos iniciales |
| `weight_max_value` | 30 | Límite superior de pesos |
| `weight_min_value` | -30 | Límite inferior de pesos |
| `conn_add_prob` | 0.2 | 20% de agregar una conexión nueva |
| `conn_delete_prob` | 0.2 | 20% de eliminar una conexión |
| `node_add_prob` | 0.2 | 20% de agregar un nodo oculto |
| `node_delete_prob` | 0.2 | 20% de eliminar un nodo oculto |
| `compatibility_threshold` | 3.0 | Umbral de especiación |
| `max_stagnation` | 30 | Generaciones sin mejora antes de extinguir especie |
| `species_elitism` | 1 | Al menos 1 especie siempre sobrevive |
| `elitism` | 2 | Los 2 mejores de cada especie pasan sin mutación |
| `survival_threshold` | 0.2 | Solo el 20% mejor de cada especie se reproduce |
| `fitness_criterion` | `max` | NEAT maximiza el fitness |
| `no_fitness_termination` | `True` | Entrena hasta cierre manual |

### 5.3 Evolución de la topología

Al inicio, cada genoma tiene solo 30 entradas → 4 salidas con conexiones aleatorias. No tiene neuronas ocultas.

Durante la evolución, **NEAT puede agregar o eliminar**:

- **Neuronas ocultas**: cuando se agrega una neurona, una conexión existente se divide en dos (entrada → neurona → salida), preservando el peso original
- **Conexiones**: se crean nuevas rutas entre nodos existentes
- **Funciones de activación**: pueden mutar entre `tanh` y `relu`

Esta capacidad de **crecer la arquitectura** es la innovación clave de NEAT: la red se adapta estructuralmente a la complejidad del problema.

### 5.4 Selección de la acción

La red produce 4 valores de salida, uno por dirección. La dirección ejecutada es la del **valor más alto** (argmax), pero solo después de pasar por el sistema de seguridad (ver sección 6).

---

## 6. Sistema de movimiento con respaldo

### 6.1 Motivación

Las redes neuronales jóvenes (primeras generaciones) toman decisiones casi aleatorias que frecuentemente son suicidas (chocar contra paredes o contra el propio cuerpo). Para evitar que la IA muera por errores triviales y pueda acumular experiencia, implementamos un **sistema jerárquico de 5 niveles de fallback**.

### 6.2 Los 5 niveles

```
NIVEL 1 — NEAT (decisión principal)
  La red neuronal elige una dirección.
  Se filtra: no puede ser giro de 180° (reversa instantánea).
  Si es segura (no choca contra cuerpo o pared) → se ejecuta.

  │
  ▼ (si falla)

NIVEL 2 — Zigzag boustrophedon (ciclo Hamiltoniano)
  Patrón fila a fila: filas pares → derecha, filas impares → izquierda.
  Al final de cada fila, baja una fila.
  En la última fila jugable, sube por la columna 1.
  Garantiza recorrer todo el tablero sin colisionar consigo misma.

  │
  ▼ (si falla)

NIVEL 3 — Flood Fill (BFS)
  Para cada dirección candidata, ejecuta BFS desde esa celda.
  Cuenta cuántas celdas vacías son alcanzables.
  Elige la dirección que maximiza el espacio libre futuro.

  │
  ▼ (si falla)

NIVEL 4 — Cualquier dirección segura
  Incluye el giro de 180° (antes prohibido).
  Última oportunidad antes de la muerte inevitable.

  │
  ▼ (si falla)

NIVEL 5 — Sin filtro
  Todas las direcciones llevan a la muerte.
  La red elige libremente (al menos la dirección puede ser útil
  para que NEAT aprenda de la penalización).
```

### 6.3 Importancia del sistema

Este sistema actúa como un **colchón de seguridad**. En operación normal, NEAT decide en el 95%+ de los casos. Los fallbacks solo entran en situaciones críticas donde ninguna salida de NEAT es segura. Sin este sistema, las primeras generaciones morirían en 2-3 pasos y nunca acumularían experiencia para aprender.

---

## 7. Función de fitness: cómo aprende

### 7.1 La señal de aprendizaje

El **fitness** es la única retroalimentación que recibe la IA. Es una puntuación numérica que mide qué tan bien jugó un genoma. NEAT maximiza esta puntuación a lo largo de las generaciones.

### 7.2 Componentes del fitness

Cada genoma juega `partidas_por_tam = 10` partidas. El fitness final es el **promedio** sobre todas ellas. Por cada partida, se acumulan 4 señales:

| Señal | Valor | Propósito |
|---|---|---|
| **Comida consumida** | +500.0 por pieza | Recompensa principal: el objetivo del juego |
| **Supervivencia** | +0.01 por paso | Incentivo mínimo a no morir rápidamente |
| **Shaping de distancia** | +1.0 × (dist_antes - dist_después) | Feedback denso: acercarse a la comida da puntos, alejarse resta |
| **Penalización auto-colisión** | −300.0 si elige chocar contra su cuerpo | Castigo específico a la causa de muerte más evitable |

### 7.3 Fórmula completa

```
fitness = Σ(comida × 500 + pasos × 0.01 + Σ_shaping - auto_colision × 300)
          ────────────────────────────────────────────────────────────────
                                    partidas_por_tam

Donde:
  Σ_shaping = Σ(distancia_manhattan_antes - distancia_manhattan_después)
  auto_colision = 1 si la serpiente elige moverse hacia su propio cuerpo
```

### 7.4 Ejemplo concreto

Un genoma que en sus 10 partidas logra:
- Comer 5 frutas en promedio
- Vivir 200 pasos por partida
- Acercarse consistentemente a la comida (+0.5 de shaping por paso en promedio)
- Nunca chocar contra su cuerpo

```
Fitness = (5 × 500 + 200 × 0.01 + 200 × 0.5 - 0 × 300) / 10
        = (2500 + 2 + 100) / 10
        = 260.2
```

### 7.5 Condiciones de fin de partida

| Condición | Causa | Consecuencia |
|---|---|---|
| **Muerte** | Choca contra pared o cuerpo | Termina la partida |
| **Hambre** | `timer_comida ≥ max_pasos_hambre` (320 pasos) | Termina la partida |
| **Victoria** | Tablero completamente lleno (1,600 celdas) | Bonus de +5,000 puntos y termina |

El límite de hambre se calcula como `tablero² / 5 = 1600 / 5 = 320` pasos. Se reinicia cada vez que la serpiente come.

### 7.6 Importancia del shaping de distancia

El **shaping** es fundamental para resolver el problema de señal escasa. Sin él, la red solo recibe retroalimentación cuando come (+500) o cuando muere. Al inicio del entrenamiento, los eventos de comida son extremadamente raros, por lo que el gradiente de aprendizaje es casi nulo.

El shaping proporciona **feedback denso paso a paso**: la red recibe una pequeña recompensa cada vez que se acerca a la comida, incluso si no llega a comerla. Esto acelera enormemente el aprendizaje temprano.

---

## 8. Interfaz gráfica

### 8.1 Vista general

La GUI está construida con **Pygame-CE** y corre en el **hilo principal** a 60 FPS, mientras la evolución ocurre en un hilo separado en segundo plano. La ventana tiene resolución fija de **1280×720**.

### 8.2 Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: "SNAKE NEAT" · [EVAL/VIVO] · MEJOR · FPS · ESP · GEN     │
├──────────────────────────┬──────────────────────────────────────────┤
│                          │  ┌────────────────────────────────────┐  │
│   ┌──────────────────┐   │  │  GENERACIÓN                       │  │
│   │                  │   │  │  NÚMERO (grande)                  │  │
│   │   TABLERO        │   │  │  ESPECIES · TIEMPO/GEN            │  │
│   │   DE JUEGO       │   │  ├────────────────────────────────────┤  │
│   │   40×40          │   │  │  FITNESS HISTÓRICO                │  │
│   │                  │   │  │  ┌──────────────────────────┐     │  │
│   │   La serpiente   │   │  │  │  Gráfica (60 puntos)     │     │  │
│   │   juega sola     │   │  │  │  Verde = mejor           │     │  │
│   │                  │   │  │  │  Ámbar = promedio        │     │  │
│   └──────────────────┘   │  │  └──────────────────────────┘     │  │
│                          │  ├────────────────────────────────────┤  │
│                          │  │  RENDIMIENTO · MEJOR · PROMEDIO   │  │
│                          │  ├────────────────────────────────────┤  │
│                          │  │  EN VIVO                          │  │
│                          │  │  SCORE · MEJOR · LONG · DIR       │  │
│                          │  │  MUERTES · VICTORIAS              │  │
│                          │  ├────────────────────────────────────┤  │
│                          │  │  HAMBRE: [████████░░] OK          │  │
│                          └──────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Panel izquierdo — Tablero de juego

- Muestra la serpiente del **mejor genoma de la generación actual** jugando en vivo
- Renderizado a ~25 pasos por segundo (desacoplado de los 60 FPS de la GUI)
- Utiliza el mapa de colores del tema synthwave/outrun
- Cuando termina la partida visual, reinicia automáticamente con el mismo genoma

### 8.4 Panel derecho — Estadísticas

| Sección | Contenido |
|---|---|
| **GENERACIÓN** | Número de generación actual (grande) |
| | Especies activas, Tiempo promedio por generación |
| **FITNESS HISTÓRICO** | Gráfica de línea dual (últimos 60 puntos) |
| | Verde: mejor genoma por generación |
| | Ámbar: fitness promedio de la población |
| | Etiquetas con valor máximo |
| **RENDIMIENTO** | Tarjetas: Mejor fitness histórico, Promedio actual |
| **EN VIVO** | Score actual, Mejor score histórico, Longitud de la serpiente |
| | Dirección actual (flecha), Muertes visuales, Victorias visuales |
| **HAMBRE** | Barra de progreso con color dinámico: |
| | Verde (< 50%) → Ámbar (< 80%) → Rojo (≥ 80%) |
| | Etiqueta: OK / ALERTA / CRÍTICO |

### 8.5 Botón "Ver Red Neuronal"

Genera un diagrama de la **topología exacta** del mejor genoma usando Graphviz:
- **Cuadros grises**: neuronas de entrada (30)
- **Círculos azules**: neuronas de salida (4)
- **Círculos blancos**: neuronas ocultas (creadas por NEAT)
- **Flechas verdes**: conexiones con peso positivo
- **Flechas rojas**: conexiones con peso negativo
- **Grosor de flecha**: magnitud del peso

---

## 9. Evaluación paralela y rendimiento

### 9.1 Paralelización

Para acelerar el entrenamiento, la evaluación de los 200 genomas se distribuye entre múltiples núcleos de CPU usando `multiprocessing.Pool`:

```
Con GUI activa:   num_nucleos = max(1, CPU_count // 4)
Sin GUI (headless): num_nucleos = max(1, CPU_count // 2)
```

En un procesador de 8 núcleos con GUI:
- 2 workers evalúan ~33 genomas cada uno
- Cada worker simula secuencialmente las 10 partidas de cada genoma
- Las partidas se ejecutan a velocidad máxima (sin renderizado)

### 9.2 Aceleración con Numba

El motor de juego (`fast_snake.py`) está compilado con **Numba JIT** usando el decorador `@njit(cache=True)`. Esto convierte el código Python en código máquina nativo en la primera ejecución, logrando velocidades **10-50× más rápidas** que Python puro.

Las funciones aceleradas son:
- `move_snake()` — ejecuta un paso del juego (validación, colisión, actualización)
- `get_rand_empty_pos()` — encuentra una celda vacía aleatoria para la comida
- `bresenham()` — traza los rayos de percepción celda por celda
- `lanzar_rayos()` — lanza todos los rayos y recolecta distancias
- `escanear_rayo()` — recorre un rayo detectando obstáculos

### 9.3 Tiempos estimados

| Operación | Tiempo |
|---|---|
| 1 generación (200 genomas × 10 partidas) | 10-30 segundos |
| 100 generaciones | ~20-50 minutos |
| 1,000 generaciones | ~3-8 horas |
| 6,000 generaciones | ~1-2 días de CPU continua |
| 1 partida visual en GUI | ~3-4 segundos (320 pasos máximo) |

---

## 10. Checkpoints y continuidad

### 10.1 Sistema de checkpoints

El entrenamiento puede **interrumpirse y reanudarse** en cualquier momento gracias a un sistema de checkpoints automáticos.

### 10.2 Funcionamiento

Al iniciar:
1. Escanea la carpeta de checkpoints en busca de archivos `neat-checkpoint-*`
2. Los ordena por número de generación (descendente)
3. Intenta cargar el más reciente
4. **Si está corrupto**, prueba con el anterior (fallback automático)
5. Si no hay ninguno, crea una población nueva desde Generación 0

### 10.3 Frecuencia de guardado

- Cada **25 generaciones** (automático, configurable)
- Al **cerrar la ventana** (checkpoint de emergencia)
- El nombre del archivo codifica la generación: `neat-checkpoint-1234`

### 10.4 Nomenclatura de carpetas

Cada configuración de parámetros de percepción tiene su **propia carpeta** de checkpoints para evitar mezclar estados incompatibles:

```
checkpoints-neat-snake-8-rays-True-False-True-False-1-v2/
  ├── neat-checkpoint-0
  ├── neat-checkpoint-25
  ├── neat-checkpoint-50
  ├── ...
  └── training_log.csv
```

El nombre codifica:
- `8-rays` = número de rayos
- `True` = dir_rotativas activado
- `False` = incluir longitud de serpiente desactivado
- `True` = incluir distancia a paredes activado
- `False` = incluir última dirección desactivado
- `1` = long_temporal (frames apilados)
- `v2` = versión de la arquitectura

### 10.5 Registro de métricas

El archivo `training_log.csv` registra por cada generación:

| Columna | Descripción |
|---|---|
| `generacion` | Número de generación |
| `mejor_fitness` | Fitness del mejor genoma |
| `fitness_promedio` | Fitness promedio de la población |
| `num_especies` | Cantidad de especies activas |
| `tiempo_s` | Tiempo en segundos que tomó la generación |

---

## 11. Stack tecnológico y dependencias

### 11.1 Lenguaje y librerías

| Librería | Versión mín. | Propósito |
|---|---|---|
| **Python** | 3.11+ | Lenguaje base |
| **neat-python** | 1.0+ | Implementación del algoritmo NEAT |
| **NumPy** | 1.24+ | Arrays del tablero, operaciones matriciales |
| **Numba** | 0.57+ | Compilación JIT del motor de juego y raycasting |
| **Pygame-CE** | 2.4+ | Renderizado de ventana, gráficos 2D y eventos |
| **Graphviz** | (sistema) | Diagrama de topología de red neuronal |
| **Matplotlib** | 3.7+ | Gráficas de fitness y especiación (post-entrenamiento) |

### 11.2 Archivo requirements.txt

```
pygame-ce
neat-python
numpy
numba
graphviz
matplotlib
pygame-gui
```

### 11.3 Instalación

```bash
# 1. Clonar o descargar el proyecto
# 2. Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Instalar dependencias Python
pip install -r src/requirements.txt

# 4. (Opcional) Instalar Graphviz para visualizar redes
# Windows: winget install graphviz
# Linux: sudo apt install graphviz
# Mac: brew install graphviz
```

---

## 12. Guía de ejecución

### 12.1 Iniciar entrenamiento

```bash
python src/snake_ai/train.py
```

El programa automáticamente:
1. Busca el checkpoint más reciente
2. Carga la población guardada
3. Inicia la ventana GUI (si `habilitar_gui = True`)
4. Comienza a evolucionar desde donde se quedó

### 12.2 Reiniciar desde cero

```bash
# Método 1: usando el script
python reset_evolution.py

# Método 2: manual
# Eliminar la carpeta de checkpoints correspondiente
Remove-Item -Recurse checkpoints-neat-snake-8-rays-True-False-True-False-1-v2/
```

### 12.3 Modo headless (sin ventana)

Para entrenamiento más rápido sin interfaz gráfica, editar `src/snake_ai/parametros.py`:

```python
habilitar_gui = False   # True → False
```

Esto también duplica los núcleos disponibles para evaluación paralela:
```python
# Con GUI: CPU_count // 4
# Sin GUI: CPU_count // 2
```

### 12.4 Personalización de parámetros

Los principales hiperparámetros se configuran en `src/snake_ai/parametros.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `num_rayos` | 8 | Número de rayos de percepción |
| `dir_rotativas` | True | Rayos egocéntricos (rotan con dirección) |
| `incluir_dist_pared` | True | Incluye 4 distancias a paredes |
| `incluir_ultima_dir` | False | Incluye última dirección (experimental) |
| `incluir_long_serp` | False | Incluye longitud de serpiente (experimental) |
| `partidas_por_tam` | 10 | Partidas por genoma por generación |
| `recompensa_comida` | 500.0 | Fitness por comida consumida |
| `bonus_supervivencia` | 0.01 | Fitness por paso vivido |
| `penalizacion_auto_colision` | 300.0 | Penalización por chocar con cuerpo propio |
| `intervalo_checkpoint` | 25 | Generaciones entre checkpoints |

**Importante**: cambiar `num_rayos`, `incluir_dist_pared`, `incluir_ultima_dir` o `incluir_long_serp` cambia el tamaño del vector de entrada. Esto requiere:
1. Actualizar `num_inputs` en el archivo `src/snake_ai/config`
2. Los checkpoints existentes serán incompatibles (usar una carpeta nueva)

---

## 13. Evolución del diseño (v1 → v2)

### 13.1 Versión 1 (4 rayos absolutos, 14 inputs)

La primera versión del proyecto utilizaba:
- **4 rayos cardinales** (N, S, E, O) en coordenadas absolutas
- **14 inputs** totales: `4 × 2 (proximidad + flag comida) + 4 (paredes) + 2 (vector comida)`
- Normalización lineal: `1.0 - (distancia / diagonal)`
- Sin separación entre cuerpo y pared (una sola distancia por rayo)

**Limitaciones**:
- La red aprendía patrones absolutos ("peligro al norte") que no generalizaban cuando la serpiente cambiaba de dirección
- 4 rayos perdían obstáculos en diagonales
- Un solo valor por rayo mezclaba cuerpo y pared, que tienen dinámicas diferentes

### 13.2 Versión 2 (8 rayos rotativos, 30 inputs) — Actual

La versión actual introdujo:

| Mejora | v1 | v2 | Beneficio |
|---|---|---|---|
| Rayos | 4 cardinales | 8 radiales | Mejor cobertura angular (45° vs 90°) |
| Orientación | Absoluta | Egocéntrica (rotativa) | Generaliza entre direcciones |
| Distancias por rayo | 1 (obstáculo genérico) | 2 (cuerpo + pared) | Modela amenazas distintas |
| Normalización | Lineal | Hiperbólica (1/dist) | Sensible a peligros cercanos |
| Fitness | Solo comida + pasos | + Shaping + Penalización | Mejor gradiente de aprendizaje |
| Fallback | Ninguno | Zigzag + Flood Fill | Evita muertes triviales |

### 13.3 Decisiones técnicas clave

1. **8 rayos en lugar de 4**: los obstáculos diagonales (frecuentes en un tablero cuadrado) son invisibles con solo 4 direcciones cardinales. 8 rayos (cada 45°) cubren todas las direcciones relevantes.

2. **Rayos egocéntricos**: rotar los rayos con la dirección de movimiento permite que la red aprenda "hay peligro adelante" en lugar de "hay peligro al norte". Esto **reduce la cantidad de patrones** que la red debe aprender (no necesita re-aprender el mismo patrón para cada orientación).

3. **Separación cuerpo/pared**: la pared es estática y siempre letal; el cuerpo es dinámico (cambia cada paso) y a veces puede tocarse (cuando la cola se mueve). Modelarlos por separado da a la red más información para tomar decisiones.

4. **Normalización hiperbólica**: `proximidad = 1 / max(distancia, 1)` da una curva exponencial inversa. Cuando el obstáculo está a 1 celda → proximidad = 1.0 (máximo peligro). A 2 celdas → 0.5. A 4 celdas → 0.25. La red es **muy sensible a peligros cercanos** y puede ignorar los lejanos.

5. **Shaping de distancia**: la recompensa directa por comida es muy escasa al inicio del entrenamiento. El shaping da feedback paso a paso por acercarse o alejarse de la comida, creando un **gradiente de aprendizaje continuo** incluso antes de que la serpiente aprenda a comer.

---

## 14. Problemas conocidos y bugs

### 14.1 Bug principal: desincronización en move_snake()

| Campo | Detalle |
|---|---|
| **Síntoma** | La serpiente muere sin colisionar visualmente con nada |
| **Archivo** | `lib/fast_snake/src/fast_snake/fast_snake.py` |
| **Causa** | Orden incorrecto de actualización entre `game_array` y `snake_data` |
| **Impacto** | Cuando la serpiente intenta moverse hacia donde estaba su cola, la cola se borra ANTES de que la nueva cabeza se marque, dejando el tablero en un estado inválido |

**Estado**: Parcialmente corregido. La función `move_snake()` tiene una condición especial `is_at_tail` que maneja este caso, pero puede haber casos borde no cubiertos.

### 14.2 Bug secundario: victoria incompleta

Cuando el tablero está completamente lleno (1,600 celdas), `get_rand_empty_pos()` devuelve `(-1, -1)`. Este valor se usa para detectar la victoria, pero la detección no siempre funciona correctamente en todos los flujos de la función `move_snake()`.

### 14.3 Problema de diseño: partida visual muy rápida

La GUI muestra solo una partida del mejor genoma por generación, que dura máximo **320 pasos** (~3 segundos a 25 FPS). Esto hace que la demostración visual parezca demasiado rápida. Se ha propuesto aumentar el límite de hambre o congelar la pantalla unos segundos al morir.

### 14.4 Bugs corregidos

| Bug | Descripción | Solución |
|---|---|---|
| **Falso positivo en raycasting** | Distancias inicializadas en 0 → la serpiente "veía" peligro máximo en direcciones libres | Inicializar con centinela grande (grid × 3) |
| **Hambre injusta** | Límite de 120 pasos para tablero 40×40 era insuficiente | Cambiar a `grid² / 5 = 320` pasos |
| **Máximo histórico reiniciado** | `_best_visual_score` se reseteaba cada generación en la GUI | Rastrear independientemente del ciclo de generaciones |
| **Etiqueta incorrecta en gráfica** | La gráfica de fitness mostraba la etiqueta "promedio" pero graficaba el mejor | Añadir serie dual: mejor (verde) + promedio (ámbar) |

---

## 15. Trabajo futuro

### 15.1 Correcciones pendientes

- [ ] **Arreglo definitivo de `move_snake()`**: refactorizar para usar un `GameState` inmutable que evite mutaciones accidentales y garantice consistencia entre `game_array` y `snake_data`
- [ ] **Detección robusta de victoria**: manejar correctamente el caso de tablero lleno en todos los flujos

### 15.2 Mejoras propuestas

- **Aumentar `max_pasos_hambre`**: un límite más alto permitiría estrategias de exploración más largas
- **Recompensa por distancia explorada**: añadir un bonus por cubrir nuevas áreas del tablero, incentivando la exploración
- **Redes convolucionales 1D**: en lugar de fully connected en la entrada de rayos, usar una pequeña convolución 1D sobre los 8 rayos podría ayudar a extraer patrones espaciales
- **Multi-objetivo**: maximizar comida Y distancia explorada simultáneamente usando fitness multi-objetivo
- **Congelar pantalla post-mortem**: añadir una pausa de 2-3 segundos cuando la serpiente muere en la GUI para mejor experiencia visual
- **Indicador de evaluación**: mostrar "Generación N — evaluando genoma X/200" en la GUI para que el usuario entienda que el entrenamiento continúa aunque la partida visual haya terminado

### 15.3 Experimentos futuros

- Probar con **más rayos** (16) para ver si mejora la percepción espacial
- Probar con **menos rayos** (4) para ver si la red puede aprender con información más limitada
- Cambiar la **función de activación** base a `relu` y comparar tasas de convergencia
- Habilitar `incluir_ultima_dir` y `incluir_long_serp` para dar más contexto a la red
- Probar con **diferentes tamaños de tablero** (20×20, 30×30) para ver cómo escala el aprendizaje

---

*Documento generado para entrega de proyecto — Mayo 2026*

*Snake NEAT AI — Universidad*
