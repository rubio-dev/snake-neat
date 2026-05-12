# VARIABLES GLOBALES
direcciones_lista = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Cuatro direcciones (delta_fila, delta_col): derecha, izquierda, abajo, arriba

direcciones_opuestas = { # Dirección opuesta para cada una de las cuatro direcciones
    (0,  1): (0, -1),
    (0, -1): (0,  1),
    (1,  0): (-1, 0),
    (-1, 0): (1,  0),
}

# FUNCIONES
# Devuelve la dirección de mayor puntuación descartando el giro de 180°
def elegir_direccion(salidas, ultima_dir):
    opuesta = direcciones_opuestas.get(ultima_dir)

    for i in sorted(range(len(salidas)), key = lambda i: salidas[i], reverse = True):
        direccion = direcciones_lista[i]

        if direccion != opuesta:
            return direccion

    return direcciones_lista[salidas.index(max(salidas))]
