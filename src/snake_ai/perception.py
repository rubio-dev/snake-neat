import math # Trigonometría y raíz cuadrada para raycasting y normalización
import numpy # Arrays numéricos y operaciones vectoriales

from numba import njit # Compilación JIT para acelerar las funciones de raycasting
from fast_snake.fast_snake import TileTypes # Tipos de celda del motor de juego

# FUNCIONES
# Traza una línea entera entre dos puntos usando el algoritmo de Bresenham
@njit
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

# Recorre las celdas de un rayo hasta encontrar obstáculo o comida y escribe el resultado
@njit
def escanear_rayo(indice_rayo, tablero, rayo, cabeza_x, cabeza_y, dist_obstaculos, band_comida, tam_cuad):
    for punto in rayo:
        x = int(punto[0])
        y = int(punto[1])

        # Detener si el rayo abandona los límites del tablero
        if not (0 <= x < tablero.shape[1] and 0 <= y < tablero.shape[0]):
            break

        celda = tablero[y][x]

        if celda == TileTypes.FOOD.value:
            band_comida[indice_rayo] = 1.0

            break
        elif celda == TileTypes.SNAKE_BODY.value or celda == TileTypes.WALL.value:
            dist_obstaculos[indice_rayo] = math.sqrt((x - cabeza_x) ** 2 + (y - cabeza_y) ** 2) # Distancia euclidiana cruda al obstáculo

            break

# Lanza n_rayos radiales desde la cabeza y retorna distancias a obstáculos y banderas de comida
@njit
def lanzar_rayos(serpiente, tablero, n_rayos = 8):
    tam_cuad = tablero.shape[1]
    cabeza_x = serpiente[-1][1] # Columna de la cabeza
    cabeza_y = serpiente[-1][0] # Fila de la cabeza

    # Centinela grande: rayos sin impacto producen 0.0 tras normalizar (sin peligro)
    dist_obstaculos = numpy.full(n_rayos, float(tam_cuad * 3))
    band_comida = numpy.zeros(n_rayos)

    angulo = 0.0

    for i in range(n_rayos):
        fin_x = cabeza_x + int(math.cos(angulo) * tam_cuad * 2)
        fin_y = cabeza_y + int(math.sin(angulo) * tam_cuad * 2)
        rayo = bresenham(cabeza_x, cabeza_y, fin_x, fin_y)
        escanear_rayo(i, tablero, rayo, cabeza_x, cabeza_y, dist_obstaculos, band_comida, tam_cuad)
        angulo += 2.0 * math.pi / n_rayos

    return dist_obstaculos, band_comida

# Genera el vector de observación completo para la red neuronal dado el estado del juego
def obtener_entradas(juego, n_rayos = 8, incluir_dist_pared = False, incluir_long_serp = False):
    tablero = juego[0]
    pos_comida = juego[1]
    serpiente = juego[2]

    cols_tablero = tablero.shape[1]
    filas_tablero = tablero.shape[0]

    cabeza_x = float(serpiente[-1][1])
    cabeza_y = float(serpiente[-1][0])

    # Vector normalizado hacia la comida; señal neutra si el tablero está lleno (victoria)
    if pos_comida == (-1, -1):
        dx, dy = 0.0, 0.0
    else:
        dx = (pos_comida[1] - cabeza_x) / cols_tablero
        dy = (pos_comida[0] - cabeza_y) / filas_tablero

    dist_brutas, band_comida = lanzar_rayos(serpiente, tablero, n_rayos)

    # Normalizar distancias con la diagonal real del tablero, clip a [0.0, 1.0]
    dist_maxima = math.sqrt(float(cols_tablero) ** 2 + float(filas_tablero) ** 2)
    arr_proximidades = numpy.clip(1.0 - (numpy.array(dist_brutas, dtype = numpy.float32) / dist_maxima), 0.0, 1.0)

    dist_paredes = []

    if incluir_dist_pared:
        dist_paredes = [
            cabeza_x / cols_tablero, # Borde izquierdo
            (cols_tablero - 1 - cabeza_x) / cols_tablero, # Borde derecho
            cabeza_y / filas_tablero, # Borde superior
            (filas_tablero - 1 - cabeza_y) / filas_tablero, # Borde inferior
        ]

    entradas = arr_proximidades.tolist() + list(band_comida) + dist_paredes + [dx, dy]

    if incluir_long_serp:
        entradas.append(len(serpiente) / (cols_tablero * filas_tablero))

    return entradas

# Dibuja los rayos de la IA y la línea hacia la comida sobre la superficie (sólo para depuración)
def dibujar_entradas_ia(superficie, tablero, pos_comida, pos_cabeza, entradas, n_rayos = 8):
    import pygame # Importación local: sólo necesaria para depuración visual

    tam_celda = superficie.get_width() / tablero.shape[1]
    cabeza_x, cabeza_y = pos_cabeza[0], pos_cabeza[1]
    dx_comida = pos_comida[1] - cabeza_x
    dy_comida = pos_comida[0] - cabeza_y

    # Línea dorada desde la cabeza hasta la comida
    pygame.draw.line(
        superficie,
        (255, 215, 0),
        (cabeza_x * tam_celda + tam_celda / 2, cabeza_y * tam_celda + tam_celda / 2),
        ((cabeza_x + dx_comida) * tam_celda + tam_celda / 2, (cabeza_y + dy_comida) * tam_celda + tam_celda / 2),
        max(1, int(tam_celda / 5)),
    )

    angulo = 0.0

    # Rayos con longitud proporcional a la distancia detectada
    for prox_val in entradas[:n_rayos]:
        distancia = (1.0 - prox_val) * tablero.shape[1]
        vx = math.cos(angulo)
        vy = math.sin(angulo)
        pygame.draw.line(
            superficie,
            (255, 255, 0),
            (cabeza_x * tam_celda + tam_celda / 2, cabeza_y * tam_celda + tam_celda / 2),
            ((cabeza_x + vx * distancia) * tam_celda + tam_celda / 2, (cabeza_y + vy * distancia) * tam_celda + tam_celda / 2),
            max(1, int(tam_celda / 5)),
        )
        angulo += 2.0 * math.pi / n_rayos
