# Snake NEAT AI

Este proyecto es una implementación de Inteligencia Artificial que aprende a jugar al clásico juego de la Serpiente (Snake) desde cero, utilizando el algoritmo evolutivo **NEAT** (NeuroEvolution of Augmenting Topologies).

## ¿Cómo funciona?

El sistema utiliza principios de evolución biológica combinados con redes neuronales para enseñar a la serpiente a sobrevivir y comer manzanas. El proceso se divide en cuatro pasos principales:

1. **Percepción (Los "Ojos" de la Serpiente):** 
   En cada instante, la serpiente lanza rayos visuales (como un radar) en 8 direcciones a su alrededor. Estos rayos miden la distancia hacia las paredes, hacia su propio cuerpo y detectan si hay comida en esa línea de visión. En total, la serpiente recoge 30 valores de su entorno.
   
2. **El Cerebro (Red Neuronal):** 
   Los datos visuales entran a una red neuronal. Esta red procesa la información y emite 4 señales de salida, correspondientes a las direcciones (Arriba, Abajo, Izquierda, Derecha).

3. **Acción y Seguridad:**
   La dirección con la señal más fuerte es la elegida. Antes de moverse, un sistema de seguridad evalúa la acción: si la IA decide hacer un movimiento suicida (como chocar instantáneamente contra su cuerpo), el sistema aplica algoritmos de respaldo (como *Zigzag* o *Flood Fill*) para intentar salvarla y darle más tiempo de aprendizaje.

4. **Evolución y Selección Natural:**
   - **Evaluación (Fitness):** A cada serpiente se le da una puntuación. Gana muchos puntos por comer (+500) y puntos menores por sobrevivir y acercarse a la comida. Pierde puntos si choca.
   - **Reproducción:** Se evalúa a una población de 200 serpientes por generación. Las mejores sobreviven y se "reproducen", mezclando sus redes neuronales y sufriendo mutaciones aleatorias (se añaden nuevas conexiones o neuronas). Las peores son descartadas.
   - ¡Con cada generación, las redes neuronales evolucionan y las serpientes se vuelven más inteligentes!

## Estructura del Proyecto

El proyecto está diseñado de forma modular para maximizar el rendimiento:

*   **`lib/fast_snake/`**: El motor puro del juego. Está escrito en Python pero optimizado con **Numba** para compilarse a código máquina. Esto permite que el juego se ejecute a velocidades extremadamente altas, reduciendo drásticamente el tiempo de entrenamiento de la IA.
*   **`src/snake_ai/`**: El núcleo de la inteligencia artificial. Aquí se maneja la configuración de NEAT (`config`), el ciclo de entrenamiento (`train.py`), el cálculo de la visión (`perception.py`) y la ejecución de los movimientos (`movement.py`).
*   **`src/neat_reporters/`**: Los módulos visuales. Utiliza **Pygame** para mostrar gráficamente a la mejor serpiente de cada generación jugando en tiempo real, e incluye herramientas gráficas para visualizar el progreso de la evolución y la estructura del "cerebro" de la serpiente.
*   **`checkpoints-*/`**: Aquí se guardan automáticamente los estados del entrenamiento. Permite pausar y reanudar el aprendizaje de la IA en cualquier momento sin perder el progreso.

## Tecnologías Principales

*   **Python**: Lenguaje principal.
*   **NEAT-Python**: Librería para el algoritmo neuro-evolutivo.
*   **Numba y Numpy**: Para cálculos matemáticos acelerados y simulación ultrarrápida del juego.
*   **Pygame**: Para el renderizado gráfico de la simulación.
