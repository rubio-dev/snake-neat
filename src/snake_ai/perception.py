import math # Trigonometría y raíz cuadrada para raycasting y normalización
import numpy # Arrays numéricos y operaciones vectoriales

from numba import njit # Compilación JIT para acelerar las funciones de raycasting
from fast_snake.fast_snake import TileTypes # Tipos de celda del motor de juego

# FUNCIONES
# Traza una línea entera entre dos puntos usando el algoritmo de Bresenham
@njit(cache=True)
def bresenham(x1, y1, x2, y2):
    puntos = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    if dx > dy:
        error = dx / 2.0

        while abs(x - x2) > 0.5:
            puntos.append((x, y))
            error -= dy

            if error < 0:
                y += sy
                error += dx

            x += sx
    else:
        error = dy / 2.0

        while abs(y - y2) > 0.5:
            puntos.append((x, y))
            error -= dx

            if error < 0:
                x += sx
                error += dy

            y += sy

    puntos.append((x, y))

    return numpy.array(puntos)

# Recorre un rayo y registra por separado: distancia al cuerpo propio, distancia a pared y flag de comida
@njit(cache=True)
def escanear_rayo(indice_rayo, tablero, rayo, cabeza_x, cabeza_y, dist_cuerpo, dist_pared_rayo, band_comida, tam_cuad):
    for punto in rayo:
        x = int(punto[0])
        y = int(punto[1])

        if not (0 <= x < tablero.shape[1] and 0 <= y < tablero.shape[0]):
            break

        celda = tablero[y][x]

        if celda == TileTypes.FOOD.value:
            band_comida[indice_rayo] = 1.0
            break
        elif celda == TileTypes.SNAKE_BODY.value:
            dist_cuerpo[indice_rayo] = math.sqrt((x - cabeza_x) ** 2 + (y - cabeza_y) ** 2)
            break
        elif celda == TileTypes.WALL.value:
            dist_pared_rayo[indice_rayo] = math.sqrt((x - cabeza_x) ** 2 + (y - cabeza_y) ** 2)
            break

# Lanza n_rayos radiales desde la cabeza; angulo_inicial rota todos los rayos (para dir_rotativas)
@njit(cache=True)
def lanzar_rayos(serpiente, tablero, n_rayos = 8, angulo_inicial = 0.0):
    tam_cuad = tablero.shape[1]
    cabeza_x = serpiente[-1][1] # Columna de la cabeza
    cabeza_y = serpiente[-1][0] # Fila de la cabeza

    dist_cuerpo    = numpy.full(n_rayos, float(tam_cuad * 3))
    dist_pared_rayo = numpy.full(n_rayos, float(tam_cuad * 3))
    band_comida    = numpy.zeros(n_rayos)

    angulo = angulo_inicial

    for i in range(n_rayos):
        fin_x = cabeza_x + int(math.cos(angulo) * tam_cuad * 2)
        fin_y = cabeza_y + int(math.sin(angulo) * tam_cuad * 2)
        rayo  = bresenham(cabeza_x, cabeza_y, fin_x, fin_y)
        escanear_rayo(i, tablero, rayo, cabeza_x, cabeza_y, dist_cuerpo, dist_pared_rayo, band_comida, tam_cuad)
        angulo += 2.0 * math.pi / n_rayos

    return dist_cuerpo, dist_pared_rayo, band_comida

# Genera el vector de observación completo para la red neuronal dado el estado del juego
# Layout: [dist_cuerpo × n] [dist_pared × n] [food_flag × n] [paredes × 4] [dx, dy]
def obtener_entradas(juego, n_rayos = 8, incluir_dist_pared = False, incluir_long_serp = False, dir_rotativas = False, ultima_dir = (0, 0)):
    tablero    = juego[0]
    pos_comida = juego[1]
    serpiente  = juego[2]

    cols_tablero  = tablero.shape[1]
    filas_tablero = tablero.shape[0]

    cabeza_x = float(serpiente[-1][1])
    cabeza_y = float(serpiente[-1][0])

    # Vector normalizado hacia la comida; señal neutra si el tablero está lleno (victoria)
    if pos_comida == (-1, -1):
        dx, dy = 0.0, 0.0
    else:
        dx = (pos_comida[1] - cabeza_x) / cols_tablero
        dy = (pos_comida[0] - cabeza_y) / filas_tablero

    # Rota el ángulo de inicio de los rayos según la dirección actual de movimiento
    angulo_inicial = 0.0
    if dir_rotativas and (ultima_dir[0] != 0 or ultima_dir[1] != 0):
        angulo_inicial = math.atan2(float(ultima_dir[0]), float(ultima_dir[1]))

    dist_cuerpo_brutas, dist_pared_brutas, band_comida = lanzar_rayos(serpiente, tablero, n_rayos, angulo_inicial)

    prox_cuerpo     = numpy.clip(1.0 / numpy.maximum(numpy.array(dist_cuerpo_brutas, dtype = numpy.float32), 1.0), 0.0, 1.0)
    prox_pared_rayo = numpy.clip(1.0 / numpy.maximum(numpy.array(dist_pared_brutas, dtype = numpy.float32), 1.0), 0.0, 1.0)

    dist_paredes = []

    if incluir_dist_pared:
        dist_paredes = [
            cabeza_x / cols_tablero,                     # Borde izquierdo
            (cols_tablero  - 1 - cabeza_x) / cols_tablero, # Borde derecho
            cabeza_y / filas_tablero,                    # Borde superior
            (filas_tablero - 1 - cabeza_y) / filas_tablero, # Borde inferior
        ]

    entradas = prox_cuerpo.tolist() + prox_pared_rayo.tolist() + list(band_comida) + dist_paredes + [dx, dy]

    if incluir_long_serp:
        entradas.append(len(serpiente) / (cols_tablero * filas_tablero))

    return entradas

# Dibuja los rayos de la IA y la línea hacia la comida sobre la superficie (sólo para depuración)
def dibujar_entradas_ia(superficie, tablero, pos_comida, pos_cabeza, entradas, n_rayos = 8, angulo_inicial = 0.0):
    import pygame

    tam_celda = superficie.get_width() / tablero.shape[1]
    cabeza_x, cabeza_y = pos_cabeza[0], pos_cabeza[1]
    dx_comida = pos_comida[1] - cabeza_x
    dy_comida = pos_comida[0] - cabeza_y

    pygame.draw.line(
        superficie,
        (255, 215, 0),
        (cabeza_x * tam_celda + tam_celda / 2, cabeza_y * tam_celda + tam_celda / 2),
        ((cabeza_x + dx_comida) * tam_celda + tam_celda / 2, (cabeza_y + dy_comida) * tam_celda + tam_celda / 2),
        max(1, int(tam_celda / 5)),
    )

    angulo = angulo_inicial

    # Rayos de cuerpo (rojo) y pared (gris) con longitud proporcional a la distancia
    for i in range(n_rayos):
        prox_cuerpo = entradas[i]
        prox_pared  = entradas[n_rayos + i]
        vx = math.cos(angulo)
        vy = math.sin(angulo)

        for prox_val, color in [(prox_cuerpo, (255, 80, 80)), (prox_pared, (160, 160, 160))]:
            distancia = (1.0 - prox_val) * tablero.shape[1]
            pygame.draw.line(
                superficie,
                color,
                (cabeza_x * tam_celda + tam_celda / 2, cabeza_y * tam_celda + tam_celda / 2),
                ((cabeza_x + vx * distancia) * tam_celda + tam_celda / 2, (cabeza_y + vy * distancia) * tam_celda + tam_celda / 2),
                max(1, int(tam_celda / 5)),
            )

        angulo += 2.0 * math.pi / n_rayos
