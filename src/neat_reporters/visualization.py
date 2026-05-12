import warnings # Advertencias para dependencias opcionales no instaladas
import numpy # Arrays para cálculos de estadísticas

try:
    import graphviz # Generación de diagramas SVG de redes neuronales
except ImportError:
    graphviz = None

try:
    from matplotlib import pyplot # Gráficas de fitness y especiación
except ImportError:
    pyplot = None

# FUNCIONES
# Grafica el comportamiento temporal de una neurona de disparo (modelo Izhikevich)
def graficar_disparos(disparos, mostrar = False, archivo = None, titulo = None):
    if pyplot is None:
        warnings.warn("graficar_disparos no disponible: matplotlib no instalado.", stacklevel = 2)

        return

    tiempos = [t for t, I, v, u, f in disparos]
    potenciales = [v for t, I, v, u, f in disparos]
    recuperaciones = [u for t, I, v, u, f in disparos]
    corrientes = [I for t, I, v, u, f in disparos]
    flags_disparo = [f for t, I, v, u, f in disparos]

    figura = pyplot.figure()

    pyplot.subplot(4, 1, 1)
    pyplot.ylabel("Potencial (mv)")
    pyplot.xlabel("Tiempo (ms)")
    pyplot.grid()
    pyplot.plot(tiempos, potenciales, "g-")
    pyplot.title(
        "Modelo de neurona de disparo de Izhikevich"
        if titulo is None
        else f"Modelo de neurona de Izhikevich ({titulo!s})"
    )

    pyplot.subplot(4, 1, 2)
    pyplot.ylabel("Disparo")
    pyplot.xlabel("Tiempo (ms)")
    pyplot.grid()
    pyplot.plot(tiempos, flags_disparo, "r-")

    pyplot.subplot(4, 1, 3)
    pyplot.ylabel("Recuperación (u)")
    pyplot.xlabel("Tiempo (ms)")
    pyplot.grid()
    pyplot.plot(tiempos, recuperaciones, "r-")

    pyplot.subplot(4, 1, 4)
    pyplot.ylabel("Corriente (I)")
    pyplot.xlabel("Tiempo (ms)")
    pyplot.grid()
    pyplot.plot(tiempos, corrientes, "r-o")

    if archivo is not None:
        pyplot.savefig(archivo)

    if mostrar:
        pyplot.show()
        pyplot.close()
        figura = None

    return figura

# Grafica la evolución del tamaño de cada especie a lo largo de las generaciones
def graficar_especies(estadisticas, mostrar = False, archivo = 'speciation.svg'):
    if pyplot is None:
        warnings.warn("graficar_especies no disponible: matplotlib no instalado.", stacklevel = 2)

        return

    tam_especies = estadisticas.get_species_sizes()
    num_generaciones = len(tam_especies)
    curvas = numpy.array(tam_especies).T

    figura, eje = pyplot.subplots()
    eje.stackplot(range(num_generaciones), *curvas)

    pyplot.title("Especiación")
    pyplot.ylabel("Individuos por especie")
    pyplot.xlabel("Generaciones")

    pyplot.savefig(archivo)

    if mostrar:
        pyplot.show()

    pyplot.close()

# Grafica el fitness promedio y máximo de la población a lo largo de las generaciones
def graficar_estadisticas(estadisticas, escala_log = False, mostrar = False, archivo = 'avg_fitness.svg'):
    if pyplot is None:
        warnings.warn("graficar_estadisticas no disponible: matplotlib no instalado.", stacklevel = 2)

        return

    generaciones = range(len(estadisticas.most_fit_genomes))
    mejor_fitness = [c.fitness for c in estadisticas.most_fit_genomes]
    fitness_prom = numpy.array(estadisticas.get_fitness_mean())
    desv_fitness = numpy.array(estadisticas.get_fitness_stdev())

    pyplot.plot(generaciones, fitness_prom, 'b-', label = "promedio")
    pyplot.plot(generaciones, fitness_prom - desv_fitness, 'g-.', label = "-1 std")
    pyplot.plot(generaciones, fitness_prom + desv_fitness, 'g-.', label = "+1 std")
    pyplot.plot(generaciones, mejor_fitness, 'r-', label = "mejor")

    pyplot.title("Fitness promedio y máximo de la población")
    pyplot.xlabel("Generaciones")
    pyplot.ylabel("Fitness")
    pyplot.grid()
    pyplot.legend(loc = "best")

    if escala_log:
        pyplot.gca().set_yscale('symlog')

    pyplot.savefig(archivo)

    if mostrar:
        pyplot.show()

    pyplot.close()

# Genera un diagrama Graphviz de la topología de la red neuronal de un genoma
def dibujar_red(
    config,
    genoma,
    mostrar = False,
    archivo = None,
    nombres_nodos = None,
    mostrar_inactivas = True,
    podar_nodos = False,
    colores_nodos = None,
    formato = 'svg',
):
    if graphviz is None:
        warnings.warn("dibujar_red no disponible: graphviz no instalado.", stacklevel = 2)

        return None

    if podar_nodos:
        genoma = genoma.get_pruned_copy(config.genome_config)

    if nombres_nodos is None:
        nombres_nodos = {}

    if colores_nodos is None:
        colores_nodos = {}

    diagrama = graphviz.Digraph(
        format = formato,
        node_attr = {'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'},
    )

    # Nodos de entrada: cuadrados grises
    entradas_set = set()

    for k in config.genome_config.input_keys:
        entradas_set.add(k)
        diagrama.node(
            nombres_nodos.get(k, str(k)),
            _attributes = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': colores_nodos.get(k, 'lightgray'),
            },
        )

    # Nodos de salida: círculos azules
    salidas_set = set()

    for k in config.genome_config.output_keys:
        salidas_set.add(k)
        diagrama.node(
            nombres_nodos.get(k, str(k)),
            _attributes = {
                'style': 'filled',
                'fillcolor': colores_nodos.get(k, 'lightblue'),
            },
        )

    # Nodos ocultos: círculos blancos
    for n in set(genoma.nodes.keys()):
        if n in entradas_set or n in salidas_set:
            continue

        diagrama.node(
            str(n),
            _attributes = {
                'style': 'filled',
                'fillcolor': colores_nodos.get(n, 'white'),
            },
        )

    # Conexiones: color y grosor según peso
    for conexion in genoma.connections.values():
        if conexion.enabled or mostrar_inactivas:
            nodo_entrada, nodo_salida = conexion.key
            diagrama.edge(
                nombres_nodos.get(nodo_entrada, str(nodo_entrada)),
                nombres_nodos.get(nodo_salida, str(nodo_salida)),
                _attributes = {
                    'style': 'solid' if conexion.enabled else 'dotted',
                    'color': 'green' if conexion.weight > 0 else 'red',
                    'penwidth': str(0.1 + abs(conexion.weight / 5.0)),
                },
            )

    diagrama.render(archivo, view = mostrar, engine = "dot")

    return diagrama
