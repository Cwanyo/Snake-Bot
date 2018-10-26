import random
import pygame
from Snake import Snake
from Food import Food

WINDOW_WIDTH = 30  # 30
WINDOW_HEIGHT = 30
PIXEL_SIZE = 20


class Agent:
    def __init__(self, code_id, visualization=False, fps=30):
        self.code_id = code_id
        self.snake = Snake(WINDOW_WIDTH, WINDOW_HEIGHT, PIXEL_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)
        self.food = Food(WINDOW_WIDTH, WINDOW_HEIGHT, PIXEL_SIZE)

        self.visualization = visualization
        self.window, self.clock = self.init_visualization()
        self.fps = fps

        # state infos
        self.alive = True
        self.score = 0
        self.step = 0

    def init_visualization(self):
        if self.visualization:
            window = pygame.display.set_mode((WINDOW_WIDTH * PIXEL_SIZE, WINDOW_HEIGHT * PIXEL_SIZE))
            fps = pygame.time.Clock()
            return window, fps
        else:
            return None, None

    def next_state(self, move_direction):
        self.step += 1
        print("{} | Step:{} | Score:{}".format(self.code_id, self.step, self.score))

        self.snake.change_direction(move_direction)
        self.snake.move()

        if self.snake.collision_food(self.food.location):
            self.score += 1
            self.food.state = False

        self.food.spawn(self.snake)

        if self.snake.collision_obstacles():
            print(self.code_id, "| Game Over!")
            self.alive = False

        if self.snake.get_length() == WINDOW_WIDTH * WINDOW_HEIGHT:
            print(self.code_id, "| Win!")
            self.alive = False

        if self.visualization:
            self.window.fill((0, 0, 0))
            self.food.render(self.window)
            self.snake.render(self.window)
            pygame.display.set_caption("SNAKE GAME | Step:{} | Score:{}".format(self.step, self.score))
            pygame.display.update()
            self.clock.tick(self.fps)

        return self.get_state()

    def get_state(self):
        pass


if __name__ == '__main__':
    a = Agent(1, True, 60)

    while a.alive:
        r = random.randint(-1, 1)
        a.next_state(r)
