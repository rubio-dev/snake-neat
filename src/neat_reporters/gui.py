import time # Medición de tiempos por generación
import collections # deque para buffer temporal y historial de fitness
import pathlib # Rutas para localizar recursos de datos
import threading # Lock para sincronizar acceso a mejor_genoma entre hilos

import neat # Tipos de red neuronal y BaseReporter
import numpy # Arrays para activar la red neuronal
import pygame # Ventana, eventos y renderizado 2D

from fast_snake.fast_snake import generate_game, move_snake, render # Motor de juego Snake
from neat_reporters.visualization import dibujar_red # Generación del diagrama de la red neuronal
from snake_ai.movement import elegir_direccion # Selección de dirección sin giros de 180°
from snake_ai.perception import obtener_entradas # Vector de observación para la red

# VARIABLES GLOBALES — Paleta Synthwave / Outrun
color_fondo    = (8,    4,  16)  # Fondo principal — negro violáceo
color_panel    = (12,   6,  24)  # Fondo de paneles
color_tarjeta  = (18,  10,  36)  # Interior de tarjetas
color_borde    = (100,  0, 160)  # Borde — púrpura neón
color_sep      = (28,   8,  52)  # Separadores tenues
color_acento   = (255,  0, 128)  # Rosa neón — acento principal
color_cian     = (0,  240, 255)  # Cian eléctrico — secundario
color_oro      = (255, 210,   0) # Amarillo neón — logros
color_purpura  = (180,  0, 255)  # Púrpura eléctrico
color_texto    = (220, 195, 255) # Lavanda claro — texto principal
color_apagado  = (100,  55, 145) # Etiquetas tenues
color_dim      = (12,   4,  26)  # Relleno muy oscuro
color_rojo     = (255,  50,  90) # Rojo neón — muertes

mapa_colores_tablero = {
    0:   (4,    0,   8),
    11:  (255,   0, 128),
    12:  (150,   0,  75),
    13:  (18,    6,  36),
    100: (0,   240, 255),
}

radio_esquina = 4
margen = 12
margen_mini = 8

# FUNCIONES
# Borde neon con glow simulado por capas
def rect_neon(superficie, color_bg, rect, color_borde_neon, radio = radio_esquina):
    pygame.draw.rect(superficie, color_bg, rect, border_radius = radio)
    glow = tuple(c // 5 for c in color_borde_neon)
    pygame.draw.rect(superficie, glow, rect.inflate(4, 4), 2, border_radius = radio + 2)
    pygame.draw.rect(superficie, color_borde_neon, rect, 1, border_radius = radio)

# Rectángulo simple con borde opcional
def rect_redondeado(superficie, color, rect, radio = radio_esquina, borde = None, grosor_borde = 1):
    pygame.draw.rect(superficie, color, rect, border_radius = radio)

    if borde:
        pygame.draw.rect(superficie, borde, rect, grosor_borde, border_radius = radio)

# Barra de progreso segmentada
def barra_progreso(superficie, x, y, w, h, valor, max_val, color_relleno):
    ratio = max(0.0, min(1.0, valor / max_val)) if max_val else 0.0
    num_segs = 24
    espacio = 2
    ancho_seg = max(1, (w - 2 - espacio * (num_segs - 1)) // num_segs)
    rellenos = int(ratio * num_segs)

    pygame.draw.rect(superficie, color_dim, pygame.Rect(x, y, w, h), border_radius = 2)

    for i in range(num_segs):
        bx = x + 1 + i * (ancho_seg + espacio)
        color_seg = color_relleno if i < rellenos else (24, 10, 48)
        pygame.draw.rect(superficie, color_seg, pygame.Rect(bx, y + 1, ancho_seg, h - 2))

    glow_b = tuple(c // 5 for c in color_relleno)
    pygame.draw.rect(superficie, glow_b,       pygame.Rect(x - 1, y - 1, w + 2, h + 2), 1, border_radius = 3)
    pygame.draw.rect(superficie, color_relleno, pygame.Rect(x, y, w, h), 1, border_radius = 2)

# Gráfica de línea con área sombreada y serie secundaria opcional
def mini_grafica(superficie, rect, datos, color = color_acento, datos2 = None, color2 = color_cian):
    rect_neon(superficie, color_tarjeta, rect, color_borde)

    if len(datos) < 2:
        return

    x0, y0, w, h = rect.x + 6, rect.y + 6, rect.width - 12, rect.height - 12

    todos_vals = list(datos) + (list(datos2) if datos2 and len(datos2) >= 2 else [])
    min_val, max_val = min(todos_vals), max(todos_vals)

    if min_val == max_val:
        max_val = min_val + 1

    def a_punto(i, v, n):
        return x0 + int(i / (n - 1) * w), y0 + h - int((v - min_val) / (max_val - min_val) * h)

    for i_linea in range(1, 5):
        y_linea = y0 + int(i_linea * h / 5)
        pygame.draw.line(superficie, color_sep, (x0, y_linea), (x0 + w, y_linea), 1)

    puntos = [a_punto(i, v, len(datos)) for i, v in enumerate(datos)]

    if datos2 and len(datos2) >= 2:
        puntos2 = [a_punto(i, v, len(datos2)) for i, v in enumerate(datos2)]
        pygame.draw.lines(superficie, color2, False, puntos2, 1)
        pygame.draw.circle(superficie, color2, puntos2[-1], 2)

    poligono = puntos + [(puntos[-1][0], y0 + h), (puntos[0][0], y0 + h)]
    pygame.draw.polygon(superficie, tuple(max(0, c - 175) for c in color), poligono)
    pygame.draw.lines(superficie, color, False, puntos, 2)
    pygame.draw.circle(superficie, color_texto, puntos[-1], 3)

# Grilla perspectiva estilo Outrun en un rect
def grilla_synthwave(superficie, rect, color_linea):
    x0 = rect.centerx
    y0 = rect.y
    x1, y1 = rect.x,     rect.bottom
    x2, y2 = rect.right, rect.bottom

    for i in range(9):
        t = i / 8
        pygame.draw.line(superficie, color_linea,
            (int(x1 + (x0 - x1) * t), int(y1 + (y0 - y1) * t)),
            (int(x2 + (x0 - x2) * t), int(y2 + (y0 - y2) * t)), 1)

    for j in range(1, 7):
        t = j / 6
        pygame.draw.line(superficie, color_linea,
            (int(x1 + (x0 - x1) * t), int(y1 + (y0 - y1) * t)),
            (int(x2 + (x0 - x2) * t), int(y2 + (y0 - y2) * t)), 1)

# Mini tarjeta con etiqueta arriba y valor grande centrado
def mini_tarjeta(superficie, rect, fuentes, etiqueta, valor, col_valor):
    rect_neon(superficie, color_tarjeta, rect, tuple(c // 3 for c in col_valor))
    lbl_s = fuentes['tiny'].render(etiqueta, True, color_apagado)
    val_s = fuentes['num_sm'].render(str(valor), True, col_valor)
    mid_x = rect.x + rect.width // 2
    superficie.blit(lbl_s, (mid_x - lbl_s.get_width() // 2, rect.y + 5))
    superficie.blit(val_s, (mid_x - val_s.get_width() // 2, rect.y + 5 + lbl_s.get_height() + 3))

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
        dir_rotativas = False,
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
        self.dir_rotativas = dir_rotativas
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
        self.mejor_fitness_global = 0.0
        self._lock_genoma = threading.Lock()
        self._genoma_cache = None
        self._red_cache    = None
        self._muertes_gui  = 0
        self._victorias_gui = 0
        self._mejor_score_gui = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_lock_genoma']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock_genoma = threading.Lock()

        self._genoma_cache = None
        self._red_cache = None
        self._muertes_gui = 0
        self._victorias_gui = 0
        self._mejor_score_gui = 0

    # Callbacks del ciclo NEAT (nombres en inglés requeridos por neat-python)
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
        with self._lock_genoma:
            self.mejor_genoma = mejor_genoma
        self.historial_fitness.append(mejor_genoma.fitness)
        self.historial_fitness_prom.append(self.fitness_prom)

        if mejor_genoma.fitness > self.mejor_fitness_global:
            self.mejor_fitness_global = mejor_genoma.fitness

    def _nuevo_juego(self):
        juego = generate_game(game_size = self.tam_tablero)
        s = obtener_entradas(
            juego,
            n_rayos = self.num_rayos,
            incluir_long_serp = self.incluir_long_serp,
            incluir_dist_pared = self.incluir_dist_pared,
            dir_rotativas = self.dir_rotativas,
            ultima_dir = (0, 0),
        )

        if self.incluir_ultima_dir:
            s = s + [0.0, 0.0]

        temporal = collections.deque(
            [[0.0] * len(s) for _ in range(self.long_temporal)],
            maxlen = self.long_temporal,
        )

        return juego, temporal

    def _dibujar_header(self, ventana, fuentes, fps, estado_evo):
        SW       = ventana.get_width()
        HEADER_H = 56

        pygame.draw.rect(ventana, color_panel, pygame.Rect(0, 0, SW, HEADER_H))

        # Línea neon inferior con glow
        pygame.draw.line(ventana, tuple(c // 6 for c in color_acento), (0, HEADER_H - 3), (SW, HEADER_H - 3), 1)
        pygame.draw.line(ventana, color_acento,                        (0, HEADER_H - 1), (SW, HEADER_H - 1), 1)
        pygame.draw.line(ventana, tuple(c // 6 for c in color_acento), (0, HEADER_H + 1), (SW, HEADER_H + 1), 1)

        snake_s = fuentes['titulo'].render("SNAKE ", True, color_acento)
        neat_s  = fuentes['titulo'].render("NEAT",   True, color_texto)
        ventana.blit(snake_s, (margen * 2, 14))
        ventana.blit(neat_s,  (margen * 2 + snake_s.get_width(), 14))

        evaluando = estado_evo.get('evaluando', False)
        mejor_f   = self.mejor_fitness_global

        partes = [
            ("",      "[EVAL]" if evaluando else "[VIVO]", color_oro if evaluando else color_cian),
            ("MEJOR", f"{mejor_f:,.0f}",                   color_oro),
            ("FPS",   f"{int(fps)}",                       color_cian if fps >= 55 else color_acento),
            ("ESP",   f"{self.num_especies}",              color_purpura),
            ("GEN",   f"{self.generacion:,}",              color_acento),
        ]

        sep_s  = fuentes['tiny'].render("  |  ", True, color_sep)
        x_curr = SW - margen * 2

        for etq, val, col in partes:
            val_s = fuentes['num_header'].render(val, True, col)
            x_curr -= val_s.get_width()
            ventana.blit(val_s, (x_curr, 22))

            if etq:
                etq_s = fuentes['tiny'].render(etq, True, color_apagado)
                ventana.blit(etq_s, (x_curr, 9))

            x_curr -= sep_s.get_width()
            ventana.blit(sep_s, (x_curr, 22))

    def _dibujar_panel_stats(self, superficie, panel_rect, fuentes, timer_comida, visual_score, long_serp, ultima_dir):
        x = panel_rect.x + margen
        y = panel_rect.y + margen
        w = panel_rect.width - margen * 2

        def seccion(ry, titulo, col = color_acento):
            t = fuentes['hdr'].render(titulo, True, col)
            superficie.blit(t, (x, ry))
            lx = x + t.get_width() + 8
            pygame.draw.line(superficie, color_sep, (lx, ry + t.get_height() // 2), (x + w, ry + t.get_height() // 2), 1)
            return t.get_height() + 6

        # ── GENERACIÓN ─────────────────────────────────────────────
        y += seccion(y, "GENERACION")
        gen_s = fuentes['num_xl'].render(f"{self.generacion:,}", True, color_cian)
        superficie.blit(gen_s, (x + w // 2 - gen_s.get_width() // 2, y))
        y += gen_s.get_height() + 6

        # Fila de métricas: ESPECIES · T/GEN · FPS
        meta_items = [
            (f"{self.num_especies}",         "ESPECIES", color_purpura),
            (f"{self.tiempo_prom_gen:.1f}s", "T / GEN",  color_texto),
        ]
        col_w = w // len(meta_items)

        for i_m, (val_m, lbl_m, col_m) in enumerate(meta_items):
            cx = x + i_m * col_w + col_w // 2
            v_s = fuentes['num_sm'].render(val_m, True, col_m)
            l_s = fuentes['tiny'].render(lbl_m,  True, color_apagado)
            superficie.blit(v_s, (cx - v_s.get_width() // 2, y))
            superficie.blit(l_s, (cx - l_s.get_width() // 2, y + v_s.get_height() + 2))

        y += fuentes['num_sm'].size("0")[1] + fuentes['tiny'].size("0")[1] + 16

        # ── FITNESS HISTÓRICO ──────────────────────────────────────
        y += seccion(y, "FITNESS HISTORICO", color_cian)
        graf_h = 100
        g_rect = pygame.Rect(x, y, w, graf_h)
        mini_grafica(superficie, g_rect, self.historial_fitness, datos2 = self.historial_fitness_prom)

        if self.historial_fitness:
            pico_s = fuentes['tiny'].render(f"mejor  {max(self.historial_fitness):,.0f}", True, color_acento)
            prom_s = fuentes['tiny'].render(f"prom   {self.fitness_prom:,.0f}",           True, color_cian)
            superficie.blit(pico_s, (x + w - pico_s.get_width() - 6, y + 5))
            superficie.blit(prom_s, (x + 6,                           y + 5))

        y += graf_h + 12

        # ── RENDIMIENTO: 2 tarjetas ────────────────────────────────
        y += seccion(y, "RENDIMIENTO", color_oro)
        mejor_f = self.mejor_genoma.fitness if self.mejor_genoma else 0
        card_h  = 50
        col_w_r = w // 2

        mini_tarjeta(superficie, pygame.Rect(x,               y, col_w_r - 4, card_h), fuentes, "MEJOR",    f"{mejor_f:,.0f}",          color_oro)
        mini_tarjeta(superficie, pygame.Rect(x + col_w_r, y, col_w_r - 4, card_h), fuentes, "PROMEDIO", f"{self.fitness_prom:,.0f}", color_cian)
        y += card_h + 14

        # ── EN VIVO: grid 3x2 de mini tarjetas ────────────────────
        y += seccion(y, "EN VIVO", color_purpura)

        mapa_dir = {(0, 1): "->DER", (0, -1): "<-IZQ", (1, 0): "v ABA", (-1, 0): "^ ARR"}

        live_grid = [
            ("SCORE",    str(visual_score),             color_acento),
            ("MEJOR",    str(self._mejor_score_gui),    color_oro),
            ("LONGITUD", str(long_serp),                color_texto),
            ("DIRECCION", mapa_dir.get(ultima_dir, "--"), color_cian),
            ("MUERTES",  str(self._muertes_gui),        color_rojo),
            ("VICTORIAS", str(self._victorias_gui),     color_oro),
        ]

        cols_grid = 3
        gap       = 5
        card_w    = (w - gap * (cols_grid - 1)) // cols_grid
        card_h_sm = 46

        for idx, (lbl_g, val_g, col_g) in enumerate(live_grid):
            fila  = idx // cols_grid
            col_i = idx % cols_grid
            rx = x + col_i * (card_w + gap)
            ry = y + fila * (card_h_sm + gap)
            mini_tarjeta(superficie, pygame.Rect(rx, ry, card_w, card_h_sm), fuentes, lbl_g, val_g, col_g)

        y += 2 * (card_h_sm + gap) + 8

        # ── HAMBRE ─────────────────────────────────────────────────
        ratio_hambre = min(1.0, timer_comida / self.max_pasos_hambre) if self.max_pasos_hambre else 0.0
        color_hambre = color_cian if ratio_hambre < 0.5 else (color_oro if ratio_hambre < 0.8 else color_rojo)
        badge        = "OK" if ratio_hambre < 0.5 else ("ALERTA" if ratio_hambre < 0.8 else "CRITICO")

        l_s = fuentes['tiny'].render("HAMBRE", True, color_apagado)
        b_s = fuentes['tiny'].render(badge,    True, color_hambre)
        superficie.blit(l_s, (x + margen_mini,                       y))
        superficie.blit(b_s, (x + w - b_s.get_width() - margen_mini, y))
        y += l_s.get_height() + 4
        barra_progreso(superficie, x, y, w, 12, timer_comida, self.max_pasos_hambre, color_hambre)

    def _impl_bucle_principal(self, estado_evo):
        ruta_recursos = pathlib.Path(__file__).parent.parent.parent / 'data'

        pygame.init()
        pygame.display.set_caption('Snake NEAT')

        SW, SH = 1280, 720
        ventana = pygame.display.set_mode((SW, SH))
        reloj   = pygame.time.Clock()

        ruta_raleway = ruta_recursos / 'fonts' / 'Raleway-Regular.ttf'

        def _f(size, bold = False, raleway = False):
            if raleway and ruta_raleway.exists():
                return pygame.font.Font(str(ruta_raleway), size)
            return pygame.font.SysFont('Consolas', size, bold = bold)

        fuentes = {
            'titulo':     _f(24, bold = True),
            'num_xl':     _f(52, bold = True),
            'num_header': _f(18, bold = True),
            'num_sm':     _f(18, bold = True),
            'hdr':        _f(10, bold = True, raleway = True),
            'lbl':        _f(11, raleway = True),
            'val':        _f(12, bold = True),
            'tiny':       _f(10),
        }

        HEADER_H  = 56
        panel_y   = HEADER_H + 4
        panel_h   = SH - panel_y - margen
        ancho_izq = int(SW * 0.44) - margen
        ancho_der = SW - ancho_izq - margen * 3

        panel_juego = pygame.Rect(margen,                  panel_y, ancho_izq, panel_h)
        panel_stats = pygame.Rect(ancho_izq + margen * 2, panel_y, ancho_der, panel_h)

        gs      = min(panel_juego.width - 32, panel_juego.height - 32)
        gs_rect = pygame.Rect(0, 0, gs, gs)
        gs_rect.center = panel_juego.center
        gs_surf = pygame.Surface((gs, gs))

        juego, entradas_temporales = self._nuevo_juego()
        datos_serp     = juego[2]
        ultima_dir     = (0, 0)
        timer_comida   = 0
        visual_score   = 0
        seg_por_paso   = 0.02
        max_pasos_frame = 4
        acumulador     = 0.0
        genoma_visual  = None
        ultima_gen_mostrada = self.generacion

        while estado_evo['ejecutando']:
            delta = reloj.tick(self.fps_limite) / 1000.0

            if self.generacion != ultima_gen_mostrada:
                ultima_gen_mostrada = self.generacion

            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    estado_evo['ejecutando'] = False

            with self._lock_genoma:
                genoma_actual = self.mejor_genoma

            if genoma_actual and genoma_visual is None:
                genoma_visual = genoma_actual
                juego, entradas_temporales = self._nuevo_juego()
                datos_serp   = juego[2]
                ultima_dir   = (0, 0)
                timer_comida = 0
                visual_score = 0
                acumulador   = 0.0

            acumulador += delta
            acumulador  = min(acumulador, seg_por_paso * max_pasos_frame)
            pasos_hechos = 0

            while genoma_actual and acumulador >= seg_por_paso and pasos_hechos < max_pasos_frame:
                acumulador   -= seg_por_paso
                pasos_hechos += 1

                entradas = obtener_entradas(
                    juego,
                    n_rayos            = self.num_rayos,
                    incluir_long_serp  = self.incluir_long_serp,
                    incluir_dist_pared = self.incluir_dist_pared,
                    dir_rotativas      = self.dir_rotativas,
                    ultima_dir         = ultima_dir,
                )

                if self.incluir_ultima_dir:
                    entradas = entradas + list(ultima_dir)

                entradas_temporales.append(entradas)

                if genoma_visual is not self._genoma_cache:
                    self._red_cache    = neat.nn.FeedForwardNetwork.create(genoma_visual, self.config)
                    self._genoma_cache = genoma_visual

                salidas    = self._red_cache.activate(numpy.array(entradas_temporales).flatten())
                direccion  = elegir_direccion(salidas, ultima_dir, juego[0], juego[2][-1])
                ultima_dir = direccion

                datos_serp, muerto, pos_comida, comio = move_snake(juego[0], juego[2], direccion, juego[1])
                juego = (juego[0], pos_comida, datos_serp)

                if comio:
                    visual_score += 1
                    timer_comida  = 0
                    if visual_score > self._mejor_score_gui:
                        self._mejor_score_gui = visual_score
                elif not muerto:
                    timer_comida += 1

                if timer_comida >= self.max_pasos_hambre:
                    muerto = True

                if pos_comida == (-1, -1):
                    self._victorias_gui += 1
                    muerto = True

                if muerto:
                    if pos_comida != (-1, -1):
                        self._muertes_gui += 1

                    juego, entradas_temporales = self._nuevo_juego()
                    datos_serp   = juego[2]
                    ultima_dir   = (0, 0)
                    timer_comida = 0
                    visual_score = 0
                    with self._lock_genoma:
                        genoma_visual = self.mejor_genoma
                    acumulador   = 0.0

                    break

            # RENDERIZADO
            ventana.fill(color_fondo)

            for gy in range(0, SH, 40):
                pygame.draw.line(ventana, color_sep, (0, gy), (SW, gy), 1)
            for gx in range(0, SW, 40):
                pygame.draw.line(ventana, color_sep, (gx, 0), (gx, SH), 1)

            self._dibujar_header(ventana, fuentes, reloj.get_fps(), estado_evo)

            rect_neon(ventana, color_panel, panel_juego, color_borde)

            franja_h = panel_juego.bottom - gs_rect.bottom - 8
            if franja_h > 10:
                grilla_synthwave(
                    ventana,
                    pygame.Rect(panel_juego.x + 8, gs_rect.bottom + 8, panel_juego.width - 16, franja_h - 8),
                    color_sep,
                )

            gs_surf.fill((0, 0, 0))
            render(gs_surf, juego[0], color_map_input = mapa_colores_tablero)
            ventana.blit(gs_surf, gs_rect)

            lbl_estado = fuentes['hdr'].render(
                ">> EVALUANDO" if estado_evo.get('evaluando', False) else ">> EN VIVO",
                True,
                color_oro if estado_evo.get('evaluando', False) else color_cian,
            )
            ventana.blit(lbl_estado, (panel_juego.x + 10, panel_juego.y + 10))

            rect_neon(ventana, color_panel, panel_stats, color_borde)
            self._dibujar_panel_stats(ventana, panel_stats, fuentes, timer_comida, visual_score, len(datos_serp), ultima_dir)

            pygame.display.flip()

        pygame.quit()

    def bucle_principal(self, estado_evo):
        try:
            self._impl_bucle_principal(estado_evo)
        except Exception as e:
            print(f"\n[ERROR GUI] {e}")
            import traceback
            traceback.print_exc()
