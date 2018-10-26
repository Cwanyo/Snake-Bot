import pygame
from Snake import Snake
from Food import Food

WINDOW_WIDTH = 20  # 30
WINDOW_HEIGHT = 20
PIXEL_SIZE = 20


def main():
    window = pygame.display.set_mode((WINDOW_WIDTH * PIXEL_SIZE, WINDOW_HEIGHT * PIXEL_SIZE))
    pygame.display.set_caption("SNAKE GAME")
    fps = pygame.time.Clock()

    score = 0
    snake = Snake(WINDOW_WIDTH, WINDOW_HEIGHT, PIXEL_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)
    food = Food(WINDOW_WIDTH, WINDOW_HEIGHT, PIXEL_SIZE)

    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.change_direction(snake.moves[0])
                    break
                elif event.key == pygame.K_DOWN:
                    snake.change_direction(snake.moves[1])
                    break
                elif event.key == pygame.K_LEFT:
                    snake.change_direction(snake.moves[2])
                    break
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction(snake.moves[3])
                    break

        window.fill((0, 0, 0))

        snake.move()

        if snake.collision_food(food.location):
            score += 1
            food.state = False

        food.spawn(snake)

        if snake.collision_obstacles():
            print('over')
            run = False

        if snake.get_length() == WINDOW_WIDTH * WINDOW_HEIGHT:
            print('win')
            run = False

        food.render(window)
        snake.render(window)
        pygame.display.set_caption("SNAKE GAME | Score: " + str(score))
        pygame.display.update()

        fps.tick(5)


if __name__ == '__main__':
    main()
