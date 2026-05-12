import pygame
from fast_snake.fast_snake import generate_game, render, move_snake

if __name__ == '__main__':
    game, food_pos, snake_data = generate_game(game_size=(40, 40))
    # setup pygame
    pygame.init()
    screen = pygame.display.set_mode((320, 320))
    pygame.display.set_caption('Fast Snake')
    clock = pygame.time.Clock()
    # main loop
    running = True
    direction = (0, 0)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    direction = (-1, 0)
                elif event.key == pygame.K_s:
                    direction = (1, 0)
                elif event.key == pygame.K_a:
                    direction = (0, -1)
                elif event.key == pygame.K_d:
                    direction = (0, 1)
            # move snake_data
        screen.fill((0, 0, 0))
        snake_data, is_dead, food_pos, eaten = move_snake(game, snake_data, direction, food_pos)
        render(screen, game)
        pygame.display.flip()
        clock.tick(10)
