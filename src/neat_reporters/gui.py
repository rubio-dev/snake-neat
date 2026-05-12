import time # Medición de tiempos por generación
import collections # deque para buffer temporal y historial de fitness
import pathlib # Rutas para localizar recursos de datos

import neat # Tipos de red neuronal y BaseReporter
import numpy # Arrays para activar la red neuronal
import pygame # Ventana, eventos y renderizado 2D
import pygame_gui # Botones y gestor de UI con temas

from fast_snake.fast_snake import generate_game, move_snake, render # Motor de juego Snake
from neat_reporters.visualization import dibujar_red # Generación del diagrama de la red neuronal
from snake_ai.movement import elegir_direccion # Selección de dirección sin giros de 180°
from snake_ai.perception import obtener_entradas # Vector de observación para la red

# VARIABLES GLOBALES
color_fondo = (5, 8, 5) # Negro terminal
color_panel = (10, 16, 10) # Verde militar oscuro
color_tarjeta = (14, 22, 14) # Panel ligeramente más claro
color_borde = (0, 140, 0) # Borde verde
color_acento = (0, 255, 65) # Verde fósforo (terminal bright)
color_verde = (57, 255, 20) # Verde neón
color_amarillo = (255, 176, 0) # Ámbar
color_rojo = (255, 40, 40) # Rojo caliente
color_texto = (160, 255, 160) # Fósforo tenue
color_apagado = (55, 120, 55) # Etiquetas verdes oscuras
color_tenue = (25, 45, 25) # Separadores muy tenues

mapa_colores_tablero = { # Paleta retro para el tablero de juego
    0:   (0,   0,   0),
    11:  (0,   255, 65),
    12:  (0,   160, 40),
    13:  (35,  45,  35),
    100: (255, 176, 0),
}

radio_esquina = 0 # Sin esquinas redondeadas (estilo retro cuadrado)
margen = 16 # Margen exterior entre paneles
margen_mini = 8 # Margen interior dentro de tarjetas

# FUNCIONES
# Dibuja un rectángulo con borde opcional
def rect_redondeado(superficie, color, rect, radio = radio_esquina, borde = None, grosor_borde = 1):
    pygame.draw.rect(superficie, color, rect)

    if borde:
        pygame.draw.rect(superficie, borde, rect, grosor_borde)

# Dibuja una barra de progreso horizontal segmentada
def barra_progreso(superficie, x, y, w, h, valor, max_val, color_relleno):
    ratio = max(0.0, min(1.0, valor / max_val)) if max_val else 0.0
    num_segs = 20
    espacio = 2
    ancho_seg = max(1, (w - 2 - espacio * (num_segs - 1)) // num_segs)
    rellenos = int(ratio * num_segs)

    pygame.draw.rect(superficie, (8, 14, 8), pygame.Rect(x, y, w, h))

    for i in range(num_segs):
        bx = x + 1 + i * (ancho_seg + espacio)
        color_seg = color_relleno if i < rellenos else (18, 32, 18)
        pygame.draw.rect(superficie, color_seg, pygame.Rect(bx, y + 1, ancho_seg, h - 2))

    pygame.draw.rect(superficie, color_borde, pygame.Rect(x, y, w, h), 1)

# Dibuja una gráfica de línea compacta con área sombreada; acepta una segunda serie opcional
def mini_grafica(superficie, rect, datos, color = color_acento, datos2 = None, color2 = color_amarillo):
    if len(datos) < 2:
        rect_redondeado(superficie, color_tarjeta, rect, borde = color_borde)

        return

    rect_redondeado(superficie, color_tarjeta, rect, borde = color_borde)
    x0, y0, w, h = rect.x + 4, rect.y + 4, rect.width - 8, rect.height - 8

    # Escalar ambas series en el mismo rango Y para que sean comparables
    todos_vals = list(datos) + (list(datos2) if datos2 and len(datos2) >= 2 else [])
    min_val, max_val = min(todos_vals), max(todos_vals)

    if min_val == max_val:
        max_val = min_val + 1

    def a_punto(i, v, n):
        return x0 + int(i / (n - 1) * w), y0 + h - int((v - min_val) / (max_val - min_val) * h)

    puntos = [a_punto(i, v, len(datos)) for i, v in enumerate(datos)]

    # Líneas de grilla estilo osciloscopio
    for i_linea in range(1, 4):
        y_linea = y0 + int(i_linea * h / 4)
        pygame.draw.line(superficie, color_tenue, (x0, y_linea), (x0 + w, y_linea), 1)

    pygame.draw.line(superficie, color_tenue, (x0, y0 + h), (x0 + w, y0 + h), 1)

    # Serie secundaria (promedio) dibujada primero para quedar debajo
    if datos2 and len(datos2) >= 2:
        puntos2 = [a_punto(i, v, len(datos2)) for i, v in enumerate(datos2)]
        pygame.draw.lines(superficie, color2, False, puntos2, 1)
        pygame.draw.circle(superficie, color2, puntos2[-1], 2)

    # Serie principal (mejor) con área sombreada
    poligono = puntos + [(puntos[-1][0], y0 + h), (puntos[0][0], y0 + h)]
    pygame.draw.polygon(superficie, tuple(max(0, c - 160) for c in color), poligono)
    pygame.draw.lines(superficie, color, False, puntos, 2)
    pygame.draw.circle(superficie, color_texto, puntos[-1], 3)

# CLASES
class ReportadorPygame(neat.reporting.BaseReporter):
    def __init__(
        self,
        config,
        estadisticas,
        num_rayos = 4,
        tam_tablero = (40, 40),
        incluir_ultima_dir = False,
        incluir_dist_pared = False,
        incluir_long_serp = False,
        long_temporal = 1,
        max_pasos_hambre = None,
        fps_limite = 60,
    ):
        self.config = config
        self.estadisticas = estadisticas
        self.num_rayos = num_rayos
        self.tam_tablero = tam_tablero
        self.incluir_ultima_dir = incluir_ultima_dir
        self.incluir_dist_pared = incluir_dist_pared
        self.incluir_long_serp = incluir_long_serp
        self.long_temporal = long_temporal
        self.max_pasos_hambre = max_pasos_hambre if max_pasos_hambre is not None else int(tam_tablero[0] ** 2 // 4)
        self.fps_limite = fps_limite

        self.generacion = 0
        self.mejor_genoma = None
        self.fitness_prom = 0.0
        self.num_especies = 0
        self.tiempo_prom_gen = 0.0
        self.tiempo_ini_gen = time.time()
        self.tiempos_gen = []
        self.historial_fitness = collections.deque(maxlen = 60)
        self.historial_fitness_prom = collections.deque(maxlen = 60)

        self._genoma_cache = None
        self._red_cache = None

        self._muertes_gui = 0
        self._victorias_gui = 0
        self._mejor_score_gui = 0

    # Callbacks del ciclo NEAT (nombres en inglés requeridos por la interfaz de neat-python)
    def start_generation(self, generation):
        self.generacion = generation
        self.tiempo_ini_gen = time.time()

    def end_generation(self, config, population, species_set):
        tiempo_val = time.time() - self.tiempo_ini_gen
        self.tiempos_gen.append(tiempo_val)
        self.tiempos_gen = self.tiempos_gen[-10:]
        self.tiempo_prom_gen = sum(self.tiempos_gen) / len(self.tiempos_gen)
        self.num_especies = len(species_set.species)

    def post_evaluate(self, config, population, species, mejor_genoma):
        aptitudes = [c.fitness for c in population.values()]
        self.fitness_prom = sum(aptitudes) / len(aptitudes) if aptitudes else 0.0
        self.mejor_genoma = mejor_genoma
        self.historial_fitness.append(mejor_genoma.fitness)
        self.historial_fitness_prom.append(self.fitness_prom)

    # Crea un juego nuevo e inicializa el buffer temporal a ceros
    def _nuevo_juego(self):
        juego = generate_game(game_size = self.tam_tablero)
        s = obtener_entradas(
            juego,
            n_rayos = self.num_rayos,
            incluir_long_serp = self.incluir_long_serp,
            incluir_dist_pared = self.incluir_dist_pared,
        )

        if self.incluir_ultima_dir:
            s = s + [0.0, 0.0]

        temporal = collections.deque(
            [[0.0] * len(s) for _ in range(self.long_temporal)],
            maxlen = self.long_temporal,
        )

        return juego, temporal

    # Dibuja el panel derecho completo: estadísticas, gráfica de fitness y estado en vivo
    def _dibujar_panel_stats(self, superficie, panel_rect, fuentes, timer_comida, visual_score, long_serp, ultima_dir, reloj):
        f_hdr = fuentes['hdr']
        f_eti = fuentes['lbl']
        f_val = fuentes['val']
        f_mini = fuentes['tiny']

        x = panel_rect.x + margen
        y = panel_rect.y + margen
        w = panel_rect.width - margen * 2
        reserva_btn = 58

        # Helper: fila "Etiqueta ··· Valor"
        def fila(ry, etiqueta, valor, color_v = color_texto, alto = 20):
            superficie.blit(f_eti.render(etiqueta, True, color_apagado), (x + margen_mini, ry))
            val_s = f_val.render(str(valor), True, color_v)
            superficie.blit(val_s, (x + w - val_s.get_width() - margen_mini, ry))

            return alto

        # Helper: encabezado de sección estilo terminal
        def seccion(ry, titulo):
            t = f_hdr.render(f"-- {titulo} --", True, color_acento)
            superficie.blit(t, (x, ry))

            return t.get_height() + 4

        # Tarjeta: entrenamiento
        y += seccion(y, "ENTRENAMIENTO")
        rect_redondeado(superficie, color_tarjeta, pygame.Rect(x, y, w, 88), borde = color_borde)
        cy = y + margen_mini
        cy += fila(cy, "Generación", self.generacion, color_verde)
        cy += fila(cy, "FPS", f"{int(reloj.get_fps())}")
        cy += fila(cy, "Tiempo / gen", f"{self.tiempo_prom_gen:.2f} s", color_amarillo)
        fila(cy, "Especies", self.num_especies)
        y += 88 + 12

        # Tarjeta: gráfica de fitness histórico
        y += seccion(y, "FITNESS HISTÓRICO")
        g_rect = pygame.Rect(x, y, w, 72)
        mini_grafica(superficie, g_rect, self.historial_fitness, datos2 = self.historial_fitness_prom)
        superficie.blit(f_mini.render("mejor", True, color_acento), (x + 6, y + 4))
        superficie.blit(f_mini.render("promedio", True, color_amarillo), (x + 6 + f_mini.size("mejor")[0] + 8, y + 4))

        if self.historial_fitness:
            pico_s = f_mini.render(f"↑ {max(self.historial_fitness):.0f}", True, color_verde)
            superficie.blit(pico_s, (x + w - pico_s.get_width() - 8, y + 4))
            min_s = f_mini.render(f"↓ {min(self.historial_fitness):.0f}", True, color_apagado)
            superficie.blit(min_s, (x + 6, y + 72 - min_s.get_height() - 4))

        y += 72 + 12

        # Tarjeta: rendimiento del batch
        y += seccion(y, "RENDIMIENTO")
        rect_redondeado(superficie, color_tarjeta, pygame.Rect(x, y, w, 68), borde = color_borde)
        mejor_f = self.mejor_genoma.fitness if self.mejor_genoma else 0
        cy = y + margen_mini
        cy += fila(cy, "Mejor histórico", f"{mejor_f:,.0f}", color_verde)
        fila(cy, "Promedio actual", f"{self.fitness_prom:,.0f}", color_amarillo)
        y += 68 + 12

        # Tarjeta: estado en vivo
        restante = panel_rect.bottom - reserva_btn - y - 12
        y += seccion(y, "ESTADO EN VIVO")
        rect_redondeado(superficie, color_tarjeta, pygame.Rect(x, y, w, max(110, restante - 12)), borde = color_borde)

        ratio_hambre = min(1.0, timer_comida / self.max_pasos_hambre) if self.max_pasos_hambre else 0.0
        color_hambre = color_verde if ratio_hambre < 0.5 else (color_amarillo if ratio_hambre < 0.8 else color_rojo)
        badge_hambre = "Normal" if ratio_hambre < 0.5 else ("Alerta" if ratio_hambre < 0.8 else "Crítico")

        mapa_dir = {(0, 1): "→ Derecha", (0, -1): "← Izquierda", (1, 0): "↓ Abajo", (-1, 0): "↑ Arriba"}

        cy = y + margen_mini
        cy += fila(cy, "Score actual", visual_score, color_verde)
        cy += fila(cy, "Mejor score", self._mejor_score_gui, color_acento)
        cy += fila(cy, "Longitud", long_serp)
        cy += fila(cy, "Dirección", mapa_dir.get(ultima_dir, "—"))
        cy += fila(cy, "Muertes", self._muertes_gui, color_rojo)
        cy += fila(cy, "Victorias", self._victorias_gui, color_verde)
        cy += 4

        superficie.blit(f_eti.render("Hambre", True, color_apagado), (x + margen_mini, cy))
        badge_s = f_mini.render(badge_hambre, True, color_hambre)
        superficie.blit(badge_s, (x + w - badge_s.get_width() - margen_mini, cy))
        cy += f_eti.size("Hambre")[1] + 4
        barra_progreso(superficie, x + margen_mini, cy, w - margen_mini * 2, 12, timer_comida, self.max_pasos_hambre, color_hambre)

    # Implementación interna del loop Pygame
    def _impl_bucle_principal(self, estado_evo):
        ruta_recursos = pathlib.Path(__file__).parent.parent.parent / 'data'

        pygame.init()
        pygame.display.set_caption('Snake NEAT — Entrenamiento')

        SW, SH = 1280, 720
        ventana = pygame.display.set_mode((SW, SH))
        reloj = pygame.time.Clock()

        ruta_tema = ruta_recursos / 'gui_theme.json'
        gestor_ui = pygame_gui.UIManager((SW, SH), theme_path = str(ruta_tema)) if ruta_tema.exists() else pygame_gui.UIManager((SW, SH))

        fuentes = {
            'title': pygame.font.SysFont('Consolas', 28, bold = True),
            'sub':   pygame.font.SysFont('Consolas', 10),
            'hdr':   pygame.font.SysFont('Consolas', 10, bold = True),
            'lbl':   pygame.font.SysFont('Consolas', 11),
            'val':   pygame.font.SysFont('Consolas', 12, bold = True),
            'tiny':  pygame.font.SysFont('Consolas', 10),
        }

        TITULO_H = 66
        panel_y = TITULO_H + 8
        panel_h = SH - panel_y - margen
        ancho_izq = int(SW * 0.44) - margen
        ancho_der = SW - ancho_izq - margen * 3

        panel_juego = pygame.Rect(margen, panel_y, ancho_izq, panel_h)
        panel_stats = pygame.Rect(ancho_izq + margen * 2, panel_y, ancho_der, panel_h)

        gs = min(panel_juego.width - 48, panel_juego.height - 48)
        gs_rect = pygame.Rect(0, 0, gs, gs)
        gs_rect.center = panel_juego.center
        gs_surf = pygame.Surface((gs, gs))

        BTN_H = 36
        btn_y = panel_stats.bottom - BTN_H - 12
        btn_red = pygame_gui.elements.UIButton(
            relative_rect = pygame.Rect(panel_stats.x + margen, btn_y, panel_stats.width - margen * 2, BTN_H),
            text = 'Ver Red Neuronal',
            manager = gestor_ui,
            object_id = '#net_btn',
        )

        # Estado inicial de la simulación GUI
        juego, entradas_temporales = self._nuevo_juego()
        datos_serp = juego[2]
        ultima_dir = (0, 0)
        timer_comida = 0
        visual_score = 0
        seg_por_paso = 0.02 # 50 pasos visuales por segundo
        max_pasos_frame = 4
        acumulador = 0.0
        genoma_visual = None
        genoma_pendiente = None
        ultima_gen_mostrada = self.generacion

        # Loop principal
        while estado_evo['ejecutando']:
            delta = reloj.tick(self.fps_limite) / 1000.0

            if self.generacion != ultima_gen_mostrada:
                ultima_gen_mostrada = self.generacion
                genoma_pendiente = self.mejor_genoma

            # Procesar eventos
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    estado_evo['ejecutando'] = False

                if evento.type == pygame_gui.UI_BUTTON_PRESSED:
                    if evento.ui_element == btn_red:
                        if self.mejor_genoma:
                            try:
                                print("Generando diagrama de red neuronal...")
                                dibujar_red(self.config, self.mejor_genoma, mostrar = True, archivo = 'Digraph')
                                print("Diagrama guardado: Digraph.svg")
                            except Exception as e:
                                print(f"Error al generar diagrama: {e}")
                        else:
                            print("Aún no hay un genoma disponible.")

                gestor_ui.process_events(evento)

            # Asignar primer genoma al estar disponible
            if self.mejor_genoma and genoma_visual is None:
                genoma_visual = self.mejor_genoma
                juego, entradas_temporales = self._nuevo_juego()
                datos_serp = juego[2]
                ultima_dir = (0, 0)
                timer_comida = 0
                visual_score = 0
                acumulador = 0.0

            acumulador += delta
            acumulador = min(acumulador, seg_por_paso * max_pasos_frame)

            pasos_hechos = 0

            while self.mejor_genoma and acumulador >= seg_por_paso and pasos_hechos < max_pasos_frame:
                acumulador -= seg_por_paso
                pasos_hechos += 1

                entradas = obtener_entradas(
                    juego,
                    n_rayos = self.num_rayos,
                    incluir_long_serp = self.incluir_long_serp,
                    incluir_dist_pared = self.incluir_dist_pared,
                )

                if self.incluir_ultima_dir:
                    entradas = entradas + list(ultima_dir)

                entradas_temporales.append(entradas)

                if genoma_visual is not self._genoma_cache:
                    self._red_cache = neat.nn.FeedForwardNetwork.create(genoma_visual, self.config)
                    self._genoma_cache = genoma_visual

                salidas = self._red_cache.activate(numpy.array(entradas_temporales).flatten())
                direccion = elegir_direccion(salidas, ultima_dir)
                ultima_dir = direccion

                datos_serp, muerto, pos_comida, comio = move_snake(juego[0], juego[2], direccion, juego[1])
                juego = (juego[0], pos_comida, datos_serp)

                if comio:
                    visual_score += 1
                    timer_comida = 0

                    if visual_score > self._mejor_score_gui:
                        self._mejor_score_gui = visual_score
                elif not muerto:
                    timer_comida += 1

                if timer_comida > self.max_pasos_hambre:
                    muerto = True

                # Victoria: tablero completamente lleno
                if pos_comida == (-1, -1):
                    self._victorias_gui += 1
                    muerto = True

                if muerto:
                    if pos_comida != (-1, -1):
                        self._muertes_gui += 1

                    juego, entradas_temporales = self._nuevo_juego()
                    datos_serp = juego[2]
                    ultima_dir = (0, 0)
                    timer_comida = 0
                    visual_score = 0
                    genoma_visual = self.mejor_genoma
                    genoma_pendiente = None
                    acumulador = 0.0

                    break

            # Renderizado
            ventana.fill(color_fondo)

            # Barra de título
            titulo_s = fuentes['title'].render("Snake NEAT", True, color_acento)
            sub_s = fuentes['sub'].render(
                f"[ RAYS:{self.num_rayos}  INPUTS:{len(self.config.genome_config.input_keys)}  POP:{self.config.pop_size} ]",
                True, color_apagado,
            )
            ventana.blit(titulo_s, (SW // 2 - titulo_s.get_width() // 2, 8))
            ventana.blit(sub_s, (SW // 2 - sub_s.get_width() // 2, 46))
            pygame.draw.line(ventana, color_borde, (margen, TITULO_H - 4), (SW - margen, TITULO_H - 4), 2)
            pygame.draw.line(ventana, color_tenue, (margen, TITULO_H + 1), (SW - margen, TITULO_H + 1), 1)

            # Panel izquierdo: tablero de juego
            rect_redondeado(ventana, color_panel, panel_juego, borde = color_borde)
            lbl_estado = fuentes['hdr'].render(
                "Evaluando..." if estado_evo.get('evaluando', False) else "En vivo",
                True,
                color_amarillo if estado_evo.get('evaluando', False) else color_acento,
            )
            ventana.blit(lbl_estado, (panel_juego.x + 10, panel_juego.y + 10))
            gs_surf.fill((0, 0, 0))
            render(gs_surf, juego[0], color_map_input = mapa_colores_tablero)
            ventana.blit(gs_surf, gs_rect)

            # Panel derecho: estadísticas
            rect_redondeado(ventana, color_panel, panel_stats, borde = color_borde)
            self._dibujar_panel_stats(ventana, panel_stats, fuentes, timer_comida, visual_score, len(datos_serp), ultima_dir, reloj)

            gestor_ui.update(delta)
            gestor_ui.draw_ui(ventana)
            pygame.display.flip()

        pygame.quit()

    # Inicia el loop principal de Pygame; debe llamarse desde el hilo principal
    def bucle_principal(self, estado_evo):
        try:
            self._impl_bucle_principal(estado_evo)
        except Exception as e:
            print(f"\n[ERROR GUI] {e}")
            import traceback
            traceback.print_exc()
