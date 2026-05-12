import multiprocessing # Detección automática del número de núcleos disponibles

# VARIABLES GLOBALES

# Partidas y tablero
partidas_por_tam = 10 # Partidas por genoma por tamaño de tablero cada generación
habilitar_gui = True # False para entrenamiento headless sin ventana (más rápido)
rango_tam_tablero = [40] # Tamaños de tablero sobre los que se evalúa cada genoma
tam_tablero_gui = (40, 40) # Tamaño del tablero en la simulación visual de la GUI

# Percepción de la red (cambios aquí requieren actualizar num_inputs en el archivo config)
num_rayos = 4 # Rayos de visión radiales desde la cabeza (+2 inputs por rayo)
dir_rotativas = False # True: rayos orientados relativos a la dirección de movimiento actual
incluir_ultima_dir = False # True: añade (dx, dy) de la última dirección (+2 inputs)
incluir_long_serp = False # True: añade longitud normalizada de la serpiente (+1 input)
incluir_dist_pared = True # True: añade las 4 distancias normalizadas a las paredes (+4 inputs)
long_temporal = 1 # Frames consecutivos concatenados como entrada (1 = sin memoria temporal)

# Núcleos de CPU para evaluación paralela de genomas
if habilitar_gui:
    num_nucleos = max(1, multiprocessing.cpu_count() // 4) # Reserva CPU para el hilo de la GUI
else:
    num_nucleos = max(1, multiprocessing.cpu_count() // 2)

# Fitness
recompensa_comida = 500.0 # Señal dominante: recompensa por cada pieza de comida consumida
bonus_supervivencia = 0.01 # Incentivo mínimo por sobrevivir (debe ser menor que recompensa_comida)
max_pasos_hambre = int(tam_tablero_gui[0] ** 2 // 4) # Pasos máximos sin comer antes de morir

# Checkpoints
intervalo_checkpoint = 25 # Generaciones entre guardados de checkpoint automático
gens_parada_temprana = 0 # Generaciones sin mejora para detener el entrenamiento (0 = desactivado)

carpeta_checkpoint = ( # Nombre único por configuración; evita mezclar checkpoints incompatibles
    f'checkpoints-neat-snake'
    f'-{num_rayos}-rays'
    f'-{dir_rotativas}'
    f'-{incluir_long_serp}'
    f'-{incluir_dist_pared}'
    f'-{incluir_ultima_dir}'
    f'-{long_temporal}'
    f'-v2'
)
