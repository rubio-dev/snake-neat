import csv # Escritura de métricas de entrenamiento en formato CSV
import collections # deque para buffer temporal de entradas
import os # Rutas, variables de entorno y gestión de archivos
import shutil # Detección de ejecutables en el PATH (Graphviz)
import sys # Manipulación del sys.path para importaciones del proyecto
import time # Medición de tiempos por generación
import threading # Hilo de evolución en background

# Añadir directorios del proyecto al sys.path antes de importar módulos locales
dir_script = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(os.path.dirname(dir_script))

for ruta_rel in ['src', 'lib/fast_snake/src', 'data']:
    ruta_abs = os.path.join(dir_raiz, ruta_rel)

    if ruta_abs not in sys.path:
        sys.path.append(ruta_abs)

import neat # Framework NEAT-Python para evolución neuroevolutiva
import numpy # Arrays numéricos para activación de la red
from fast_snake.fast_snake import generate_game, move_snake # Motor de juego Snake compilado con Numba
from neat_reporters import gui # Reporter con GUI Pygame
from neat_reporters import visualization # Visualización de redes y estadísticas
from snake_ai.perception import obtener_entradas # Vector de observación para la red
from snake_ai.movement import elegir_direccion # Selección de dirección sin giros de 180°
from snake_ai.parametros import ( # Hiperparámetros de entrenamiento y configuración
    partidas_por_tam, habilitar_gui, rango_tam_tablero, tam_tablero_gui,
    num_rayos, incluir_ultima_dir, incluir_long_serp, incluir_dist_pared,
    long_temporal, num_nucleos, recompensa_comida, bonus_supervivencia,
    max_pasos_hambre, intervalo_checkpoint, gens_parada_temprana, carpeta_checkpoint,
)

# Añadir Graphviz al PATH si está instalado pero no accesible en la sesión actual
if not shutil.which('dot'):
    for ruta_gv in [r"C:\Program Files\Graphviz\bin", r"C:\Program Files (x86)\Graphviz\bin", r"C:\Graphviz\bin"]:
        if os.path.isfile(os.path.join(ruta_gv, 'dot.exe')):
            os.environ['PATH'] = ruta_gv + os.pathsep + os.environ.get('PATH', '')

            break

# FUNCIONES
# Calcula el número total de entradas que debe recibir la red según la configuración actual
def calcular_tam_entradas():
    tam = num_rayos * 2 # Distancias a obstáculos + banderas de comida por rayo

    if incluir_dist_pared:
        tam += 4

    tam += 2 # Vector (dx, dy) normalizado hacia la comida

    if incluir_ultima_dir:
        tam += 2

    if incluir_long_serp:
        tam += 1

    tam *= long_temporal

    return tam

# Evalúa el fitness de un genoma simulando varias partidas de Snake
def evaluar_genoma(genoma, config):
    red = neat.nn.FeedForwardNetwork.create(genoma, config)
    fitness_total = 0.0
    total_partidas = 0

    for tam_tablero in rango_tam_tablero:
        for _ in range(partidas_por_tam):
            muerto = False
            timer_comida = 0
            ultima_dir = (0, 0)
            fitness_partida = 0.0
            juego = generate_game(game_size = (tam_tablero, tam_tablero))

            if len(juego[2]) == 0:
                continue

            entradas_ini = obtener_entradas(
                juego,
                n_rayos = num_rayos,
                incluir_long_serp = incluir_long_serp,
                incluir_dist_pared = incluir_dist_pared,
            )

            if incluir_ultima_dir:
                entradas_ini = entradas_ini + list(ultima_dir)

            entradas_temporales = collections.deque(
                [entradas_ini[:] for _ in range(long_temporal)],
                maxlen = long_temporal,
            )

            while not muerto:
                salidas = red.activate(numpy.array(entradas_temporales).flatten())
                direccion = elegir_direccion(salidas, ultima_dir)
                ultima_dir = direccion

                datos_serp, muerto, pos_comida, comio = move_snake(juego[0], juego[2], direccion, juego[1])
                juego = (juego[0], pos_comida, datos_serp)

                # Victoria: tablero completamente lleno
                if not muerto and pos_comida == (-1, -1):
                    fitness_partida += recompensa_comida * 10.0
                    muerto = True

                    break

                if comio:
                    fitness_partida += recompensa_comida
                    timer_comida = 0
                elif not muerto:
                    fitness_partida += bonus_supervivencia
                    timer_comida += 1

                if timer_comida > max_pasos_hambre:
                    muerto = True

                if not muerto:
                    proximas_entradas = obtener_entradas(
                        juego,
                        n_rayos = num_rayos,
                        incluir_long_serp = incluir_long_serp,
                        incluir_dist_pared = incluir_dist_pared,
                    )

                    if incluir_ultima_dir:
                        proximas_entradas = proximas_entradas + list(ultima_dir)

                    entradas_temporales.append(proximas_entradas)

            fitness_total += fitness_partida
            total_partidas += 1

    return float(fitness_total / total_partidas) if total_partidas > 0 else 0.0

# Registra estadísticas de fitness en un CSV al final de cada generación
class RegistradorCSV(neat.reporting.BaseReporter):
    def __init__(self, ruta_archivo):
        self.ruta_archivo = ruta_archivo
        self.generacion_actual = 0
        self.tiempo_inicio = 0.0

        if not os.path.exists(ruta_archivo):
            with open(ruta_archivo, 'w', newline = '', encoding = 'utf-8') as archivo:
                csv.writer(archivo).writerow(['generacion', 'mejor_fitness', 'fitness_promedio', 'num_especies', 'tiempo_s'])

    def start_generation(self, generation):
        self.generacion_actual = generation
        self.tiempo_inicio = time.time()

    def post_evaluate(self, config, population, species, mejor_genoma):
        aptitudes = [g.fitness for g in population.values() if g.fitness is not None]
        fitness_prom = sum(aptitudes) / len(aptitudes) if aptitudes else 0.0
        tiempo_val = time.time() - self.tiempo_inicio

        with open(self.ruta_archivo, 'a', newline = '', encoding = 'utf-8') as archivo:
            csv.writer(archivo).writerow([
                self.generacion_actual,
                f'{mejor_genoma.fitness:.4f}',
                f'{fitness_prom:.4f}',
                len(species.species),
                f'{tiempo_val:.2f}',
            ])

StatsCSVLogger = RegistradorCSV  # Alias de compatibilidad con checkpoints anteriores

# Configura e inicia el ciclo de evolución NEAT completo
def ejecutar(ruta_config):
    # Validar que num_inputs del config coincida con la configuración de flags actual
    entradas_esperadas = calcular_tam_entradas()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        ruta_config,
    )

    if config.genome_config.num_inputs != entradas_esperadas:
        print(
            f"\n[ERROR] El config tiene num_inputs={config.genome_config.num_inputs}, "
            f"pero la configuración requiere {entradas_esperadas}.\n"
            f"  → Actualiza 'num_inputs = {entradas_esperadas}' en el archivo config."
        )
        sys.exit(1)

    # Cargar el checkpoint más reciente con fallback al anterior si está corrupto
    os.makedirs(carpeta_checkpoint, exist_ok = True)

    poblacion = None

    archivos_checkpoint = [f for f in os.listdir(carpeta_checkpoint) if f.startswith('neat-checkpoint-')]

    if archivos_checkpoint:
        lista_checkpoints = sorted(
            archivos_checkpoint,
            key = lambda x: int(x.split("-")[-1]),
            reverse = True,
        )

        for archivo_checkpoint in lista_checkpoints:
            ruta_checkpoint = os.path.join(carpeta_checkpoint, archivo_checkpoint)

            try:
                print(f"Cargando checkpoint: {archivo_checkpoint}...")
                poblacion = neat.Checkpointer.restore_checkpoint(ruta_checkpoint)
                poblacion.reporters.reporters = []
                print(f"Checkpoint '{archivo_checkpoint}' cargado.")

                break
            except (EOFError, Exception) as e:
                print(f"Error en '{archivo_checkpoint}': {e}. Probando anterior...")

    if poblacion is None:
        print("Sin checkpoints válidos. Iniciando nueva población.")
        poblacion = neat.Population(config)

    # Registrar reporteros
    poblacion.add_reporter(neat.StdOutReporter(True))
    estadisticas = neat.StatisticsReporter()
    poblacion.add_reporter(estadisticas)
    poblacion.add_reporter(neat.Checkpointer(intervalo_checkpoint, filename_prefix = f'{carpeta_checkpoint}/neat-checkpoint-'))
    poblacion.add_reporter(RegistradorCSV(os.path.join(carpeta_checkpoint, 'training_log.csv')))

    evaluador = neat.ParallelEvaluator(num_nucleos, evaluar_genoma)

    # Estado compartido entre el hilo de evolución y la GUI
    estado_evo = {'ejecutando': True, 'ganador': None, 'evaluando': False}

    # Hilo de evolución en background
    def bucle_evolucion():
        try:
            estado_parada = {'mejor': 0.0, 'estancado': 0}

            def verificar_parada(genomas, config):
                if not estado_evo['ejecutando']:
                    print(f"Deteniendo en generación {poblacion.generation}. Guardando checkpoint...")

                    try:
                        checkpoint_temp = neat.Checkpointer(filename_prefix = f"{carpeta_checkpoint}/neat-checkpoint-")
                        checkpoint_temp.save_checkpoint(config, poblacion.population, poblacion.species, poblacion.generation)
                        print("Checkpoint de cierre guardado.")
                    except Exception as e:
                        print(f"Error al guardar checkpoint: {e}")

                    sys.exit(0)

                estado_evo['evaluando'] = True
                evaluador.evaluate(genomas, config)
                estado_evo['evaluando'] = False

                if gens_parada_temprana > 0:
                    mejor_actual = max(
                        (g.fitness for _, g in genomas if g.fitness is not None),
                        default = 0.0,
                    )

                    if mejor_actual > estado_parada['mejor']:
                        estado_parada['mejor'] = mejor_actual
                        estado_parada['estancado'] = 0
                    else:
                        estado_parada['estancado'] += 1

                        if estado_parada['estancado'] >= gens_parada_temprana:
                            print(f"\n[Parada temprana] Sin mejora en {estado_parada['estancado']} generaciones.")
                            estado_evo['ejecutando'] = False

            estado_evo['ganador'] = poblacion.run(verificar_parada, 50000)

        except SystemExit:
            pass
        except Exception as e:
            print(f"Error en hilo de evolución: {e}")
            import traceback
            traceback.print_exc()
        finally:
            estado_evo['ejecutando'] = False

    # Inicializar GUI si está habilitada
    reportero = None

    if habilitar_gui:
        reportero = gui.ReportadorPygame(
            config = config,
            estadisticas = estadisticas,
            num_rayos = num_rayos,
            tam_tablero = tam_tablero_gui,
            incluir_long_serp = incluir_long_serp,
            incluir_dist_pared = incluir_dist_pared,
            incluir_ultima_dir = incluir_ultima_dir,
            long_temporal = long_temporal,
            max_pasos_hambre = max_pasos_hambre,
        )
        poblacion.add_reporter(reportero)

    hilo_evolucion = threading.Thread(target = bucle_evolucion, daemon = True)
    hilo_evolucion.start()

    # Loop principal: GUI o espera activa en modo headless
    if habilitar_gui and reportero:
        try:
            reportero.bucle_principal(estado_evo)
        except Exception as e:
            print(f"Error en GUI: {e}")
            import traceback
            traceback.print_exc()
            estado_evo['ejecutando'] = False
    else:
        try:
            while hilo_evolucion.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupción por teclado. Deteniendo...")
            estado_evo['ejecutando'] = False

    # Mostrar resultados si se encontró un ganador
    if estado_evo['ganador']:
        print('\nMejor genoma:\n{!s}'.format(estado_evo['ganador']))
        visualization.dibujar_red(config, estado_evo['ganador'], mostrar = True, archivo = 'Digraph')
        visualization.graficar_estadisticas(estadisticas, escala_log = False, mostrar = True)
        visualization.graficar_especies(estadisticas, mostrar = True)

# PUNTO DE PARTIDA
if __name__ == '__main__':
    ejecutar("src/snake_ai/config")
