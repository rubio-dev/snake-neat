from collections import deque
from fast_snake.fast_snake import TileTypes

# VARIABLES GLOBALES
direcciones_lista = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Cuatro direcciones (delta_fila, delta_col): derecha, izquierda, abajo, arriba

direcciones_opuestas = { # Dirección opuesta para cada una de las cuatro direcciones
    (0,  1): (0, -1),
    (0, -1): (0,  1),
    (1,  0): (-1, 0),
    (-1, 0): (1,  0),
}

# FUNCIONES

# ── Algoritmo 1: filtro de seguridad ─────────────────────────────────────────
def _es_segura(tablero, cabeza, direccion):
    nr = int(cabeza[0]) + direccion[0]
    nc = int(cabeza[1]) + direccion[1]
    if not (0 <= nr < tablero.shape[0] and 0 <= nc < tablero.shape[1]):
        return False
    celda = tablero[nr, nc]
    return celda != TileTypes.SNAKE_BODY.value and celda != TileTypes.WALL.value

# ── Algoritmo 2: zigzag fila a fila ──────────────────────────────────────────
# Recorre el tablero en boustrophedon: filas pares hacia la derecha,
# impares hacia la izquierda. En la última fila jugable sube por col 1
# para cerrar el ciclo Hamiltoniano.
def _calcular_direccion_zigzag(cabeza, tablero):
    row = int(cabeza[0])
    col = int(cabeza[1])
    H, W = tablero.shape

    playable_row = row - 1          # 0-indexed dentro del área jugable
    min_col      = 1
    max_col      = W - 2
    last_row     = H - 2            # última fila jugable

    if row == last_row:             # cierre de ciclo: izquierda hasta col 1, luego sube
        return (0, -1) if col > min_col else (-1, 0)

    if playable_row % 2 == 0:       # filas pares → derecha
        return (0, 1) if col < max_col else (1, 0)
    else:                           # filas impares → izquierda
        return (0, -1) if col > min_col else (1, 0)

# ── Algoritmo 3: flood fill (maximizar espacio libre) ────────────────────────
# BFS desde la celda objetivo; devuelve el número de celdas alcanzables.
# Cuanto mayor el valor, más "espacio abierto" tiene esa dirección.
def _contar_espacio_libre(tablero, nr, nc):
    H, W = tablero.shape
    visitados = set()
    cola = deque([(nr, nc)])
    bloqueados = {TileTypes.SNAKE_BODY.value, TileTypes.WALL.value, TileTypes.SNAKE_HEAD.value}

    while cola:
        r, c = cola.popleft()
        if (r, c) in visitados:
            continue
        if not (0 <= r < H and 0 <= c < W):
            continue
        if tablero[r, c] in bloqueados:
            continue
        visitados.add((r, c))
        for dr, dc in direcciones_lista:
            vecino = (r + dr, c + dc)
            if vecino not in visitados:
                cola.append(vecino)

    return len(visitados)

# ── Selector principal ────────────────────────────────────────────────────────
# Prioridad:
#   1. NEAT        – mejor dirección segura según la red.
#   2. Zigzag      – patrón fila a fila cuando NEAT está bloqueado.
#   3. Flood fill  – dirección con más espacio libre cuando las dos anteriores fallan.
#   4. Cualquier dirección segura (incluyendo reversa si no hay otra).
#   5. Sin filtro  – muerte inevitable, la red elige libremente.
def elegir_direccion(salidas, ultima_dir, tablero=None, cabeza=None):
    opuesta     = direcciones_opuestas.get(ultima_dir)
    usar_filtro = tablero is not None and cabeza is not None

    candidatas = [
        direcciones_lista[i]
        for i in sorted(range(len(salidas)), key=lambda i: salidas[i], reverse=True)
        if direcciones_lista[i] != opuesta
    ]

    if usar_filtro:
        # 1. NEAT
        for d in candidatas:
            if _es_segura(tablero, cabeza, d):
                return d

        # 2. Zigzag
        zigzag = _calcular_direccion_zigzag(cabeza, tablero)
        if zigzag != opuesta and _es_segura(tablero, cabeza, zigzag):
            return zigzag

        # 3. Flood fill — elige la dirección no-reversa con más espacio libre
        mejor_dir    = None
        mejor_espacio = -1
        for d in direcciones_lista:
            if d == opuesta or not _es_segura(tablero, cabeza, d):
                continue
            nr     = int(cabeza[0]) + d[0]
            nc     = int(cabeza[1]) + d[1]
            espacio = _contar_espacio_libre(tablero, nr, nc)
            if espacio > mejor_espacio:
                mejor_espacio = espacio
                mejor_dir     = d
        if mejor_dir:
            return mejor_dir

        # 4. Cualquier dirección segura (reversa incluida)
        for d in direcciones_lista:
            if _es_segura(tablero, cabeza, d):
                return d

    # 5. Sin salida
    return candidatas[0] if candidatas else direcciones_lista[0]
