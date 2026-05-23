import enum
import random

import numpy as np
from numba import njit


class TileTypes(enum.Enum):
    EMPTY = 0
    SNAKE_HEAD = 11
    SNAKE_BODY = 12
    WALL = 13
    FOOD = 100


@njit(cache=True)
def get_rand_empty_pos(game: [[int]]) -> (int, int):
    """
    Returns a random empty position in the game_array.
    """
    empty_positions = np.where(game == TileTypes.EMPTY.value)
    # No hay celdas vacias disponibles: tablero lleno.
    if empty_positions[0].shape[0] == 0:
        return -1, -1
    index = random.randint(0, empty_positions[0].shape[0] - 1)
    return empty_positions[0][index], empty_positions[1][index]


def generate_game(food_pos=None, snake_pos=None, game_size=(300, 300)):
    """
    Generates a game_array board with a snake_data and food.

    No usa @njit: se llama una sola vez por partida, los defaults None son
    incompatibles con la inferencia de tipos estricta de Numba, y la función
    contiene listas Python nativas.
    """
    game = np.zeros(game_size, dtype=np.int8)
    # make walls
    for y in range(game.shape[0]):
        for x in range(game.shape[1]):
            if y == 0 or y == game.shape[0] - 1 or x == 0 or x == game.shape[1] - 1:
                game[y, x] = TileTypes.WALL.value
    snake = []
    if food_pos is None:
        food_pos = get_rand_empty_pos(game)
        if food_pos != (-1, -1):
            game[food_pos[0], food_pos[1]] = TileTypes.FOOD.value
    if snake_pos is None:
        snake_pos = get_rand_empty_pos(game)
        if snake_pos != (-1, -1):
            game[snake_pos[0], snake_pos[1]] = TileTypes.SNAKE_HEAD.value
            snake.append(snake_pos)

    return game, food_pos, np.array(snake)


def render(surface: 'pygame.Surface', game_array: [[int]], color_map_input: dict = None):
    """
    Renders the game_array board to the surface.
    :param surface: pygame surface
    :param game_array: game_array board
    :param color_map_input: color map
    :return:
    """
    import pygame

    color_map = {
        TileTypes.EMPTY.value:      (15, 17, 26),    # Fondo oscuro (casi negro)
        TileTypes.SNAKE_HEAD.value: (80, 220, 100),  # Verde brillante
        TileTypes.SNAKE_BODY.value: (50, 140, 200),  # Azul medio
        TileTypes.WALL.value:       (80, 85, 100),   # Gris oscuro
        TileTypes.FOOD.value:       (255, 200, 50),  # Amarillo cálido
    }
    if color_map_input is not None:
        color_map.update(color_map_input)
    grid_size = surface.get_width() / game_array.shape[1]
    # Fondo del tablero
    surface.fill(color_map[TileTypes.EMPTY.value])
    margin = 1
    gs = int(grid_size)

    # 1. Dibujar Paredes (Solo los bordes para eficiencia)
    rows, cols = game_array.shape
    wall_color = color_map[TileTypes.WALL.value]
    
    # Paredes superior e inferior
    for x in range(cols):
        pygame.draw.rect(surface, wall_color, (int(x * grid_size) + margin, 0 + margin, max(1, gs - margin), max(1, gs - margin)))
        pygame.draw.rect(surface, wall_color, (int(x * grid_size) + margin, int((rows - 1) * grid_size) + margin, max(1, gs - margin), max(1, gs - margin)))
    
    # Paredes laterales
    for y in range(rows):
        pygame.draw.rect(surface, wall_color, (0 + margin, int(y * grid_size) + margin, max(1, gs - margin), max(1, gs - margin)))
        pygame.draw.rect(surface, wall_color, (int((cols - 1) * grid_size) + margin, int(y * grid_size) + margin, max(1, gs - margin), max(1, gs - margin)))

    # 2. Dibujar Comida y Serpiente buscando solo sus valores (mucho más rápido que 1600 iteraciones)
    # Buscamos los índices de lo que no es vacío ni pared
    items_mask = (game_array != TileTypes.EMPTY.value) & (game_array != TileTypes.WALL.value)
    item_indices = np.argwhere(items_mask)
    
    for y, x in item_indices:
        tile_type = game_array[y, x]
        pygame.draw.rect(
            surface,
            color_map[tile_type],
            (int(x * grid_size) + margin, int(y * grid_size) + margin,
             max(1, gs - margin), max(1, gs - margin))
        )


def move_snake(game_array: [[int]], snake_data: [[int]], direction, food_pos):
    """
    Moves the snake_data in the given direction.
    """
    # get head position
    head_pos = snake_data[-1]
    tail_pos = snake_data[0]
    eaten = False
    dead = False

    # get new head position
    new_head_pos = (head_pos[0] + direction[0], head_pos[1] + direction[1])

    # 1. Bounds check
    if new_head_pos[0] < 0 or new_head_pos[0] >= game_array.shape[0] \
            or new_head_pos[1] < 0 or new_head_pos[1] >= game_array.shape[1]:
        return snake_data, True, food_pos, False

    tile_at_new_head = game_array[new_head_pos]

    # 2. Collision check
    is_at_tail = (new_head_pos[0] == tail_pos[0] and new_head_pos[1] == tail_pos[1])
    
    if tile_at_new_head == TileTypes.FOOD.value:
        # EATING CASE
        # Old head becomes body cell
        game_array[head_pos[0], head_pos[1]] = TileTypes.SNAKE_BODY.value
        # New head cell
        game_array[new_head_pos] = TileTypes.SNAKE_HEAD.value
        
        # Update snake_data (grows by adding new head)
        snake_data = np.concatenate([snake_data, [new_head_pos]], axis=0)
        eaten = True
        
        # Spawn new food
        food_pos = get_rand_empty_pos(game_array)
        if food_pos != (-1, -1):
            game_array[food_pos] = TileTypes.FOOD.value
            
    elif tile_at_new_head == TileTypes.EMPTY.value or is_at_tail:
        # NORMAL MOVEMENT CASE (includes following own tail)
        # 1. Clear OLD tail in game_array (only if new head is not taking its place)
        if not is_at_tail:
            game_array[tail_pos[0], tail_pos[1]] = TileTypes.EMPTY.value
            
        # 2. Update OLD head to body (only if snake length > 1)
        if len(snake_data) > 1:
            game_array[head_pos[0], head_pos[1]] = TileTypes.SNAKE_BODY.value
        # If length is 1, the old head pos was just cleared (if not is_at_tail) 
        # or will be overwritten (if is_at_tail)
            
        # 3. Set NEW head position
        game_array[new_head_pos] = TileTypes.SNAKE_HEAD.value
        
        # 4. Update snake_data (shift coordinates)
        snake_data = np.concatenate([snake_data, [new_head_pos]], axis=0)
        snake_data = snake_data[1:]
        
    else:
        # COLLISION CASE (Wall or Body or other obstacles)
        dead = True

    return snake_data, dead, food_pos, eaten
