import time
import numpy
import math
import random
import pygame

from game.snake import Snake
from game.food import Food

WINDOW_WIDTH = 20  # 30
WINDOW_HEIGHT = 20
PIXEL_SIZE = 20


class Agent:
    def __init__(self, code_id, log=False, visualization=False, fps=60):
        self.code_id = code_id
        self.snake = Snake(WINDOW_WIDTH, WINDOW_HEIGHT, PIXEL_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)
        self.food = Food(WINDOW_WIDTH, WINDOW_HEIGHT, PIXEL_SIZE)
        self.food.spawn(self.snake)

        self.log = log
        self.visualization = visualization
        self.window, self.clock = self.init_visualization()
        self.fps = fps

        # basic infos
        self.alive = True
        self.score = 0
        self.step = 0
        # useful infos
        self.s_obstacles = self.get_surrounding_obstacles()
        self.food_angle = self.get_food_angle()
        self.food_distance = self.get_food_distance()

    def init_visualization(self):
        if self.visualization:
            window = pygame.display.set_mode((WINDOW_WIDTH * PIXEL_SIZE, WINDOW_HEIGHT * PIXEL_SIZE))
            fps = pygame.time.Clock()
            return window, fps
        else:
            return None, None

    def get_random_move(self, random_n=10):

        # real random move, 1/random_n
        if random_n is not 0 and random.randint(1, random_n) == 1:
            return random.randint(-1, 1)

        s, a, d = self.get_state()
        # random move depend on state
        ops = []
        # select move based on following the food angle and avoiding the obstacles
        if not s[0] and a < 0:
            ops.insert(-1, -1)
        if not s[1] and a == 0:
            ops.insert(-1, 0)
        if not s[2] and a > 0:
            ops.insert(-1, +1)

        # if no option
        if not ops:
            # select move based on avoiding obstacles
            if not s[0]:
                ops.insert(-1, -1)
            if not s[1]:
                ops.insert(-1, 0)
            if not s[2]:
                ops.insert(-1, +1)

            # again, if no option -> just die
            if not ops:
                return random.randint(-1, 1)
            else:
                return ops[random.randint(0, len(ops) - 1)]
        else:
            return ops[random.randint(0, len(ops) - 1)]

    def next_state(self, move_direction):
        self.step += 1
        info = 'CodeID: {} | Step: {} | Score: {}'.format(self.code_id, self.step, self.score)

        self.snake.change_direction(move_direction)
        self.snake.move()

        if self.snake.collision_food(self.food.location):
            self.score += 1
            self.food.state = False

        self.food.spawn(self.snake)

        if self.snake.collision_obstacles():
            info += ' >> Game Over!'
            self.alive = False

        if self.snake.get_length() == WINDOW_WIDTH * WINDOW_HEIGHT:
            info += ' >> Win!'
            self.alive = False

        if self.log:
            print(info)

        if self.visualization:
            self.window.fill((0, 0, 0))
            self.food.render(self.window)
            self.snake.render(self.window)
            pygame.display.set_caption(info)
            pygame.display.update()
            pygame.event.get()
            self.clock.tick(self.fps)

        return self.get_state()

    def get_state(self):
        return self.get_surrounding_obstacles(), self.get_food_angle(), self.get_food_distance()

    def get_surrounding_obstacles(self):
        # check front
        snake_head = self.snake.head
        snake_heading_direction = self.snake.heading_direction
        left = self.snake.moves[(snake_heading_direction - 1) % len(self.snake.moves)]
        front = self.snake.moves[snake_heading_direction]
        right = self.snake.moves[(snake_heading_direction + 1) % len(self.snake.moves)]
        l_location = [snake_head[0] + left[0], snake_head[1] + left[1]]
        f_location = [snake_head[0] + front[0], snake_head[1] + front[1]]
        r_location = [snake_head[0] + right[0], snake_head[1] + right[1]]

        s_locations = [l_location, f_location, r_location]
        self.s_obstacles = [0, 0, 0]

        # check wall
        for i in range(0, len(s_locations)):
            if s_locations[i][0] < 0 or s_locations[i][0] >= WINDOW_WIDTH \
                    or s_locations[i][1] < 0 or s_locations[i][1] >= WINDOW_HEIGHT:
                self.s_obstacles[i] = 1

        # check body
        for b in self.snake.body:
            if b in s_locations:
                self.s_obstacles[s_locations.index(b)] = 1

        return self.s_obstacles

    def get_food_angle(self):
        # get direction of heading
        heading_direction = numpy.array(self.snake.moves[self.snake.heading_direction])
        # get direction of food (distant)
        food_direction = numpy.array(self.food.location) - numpy.array(self.snake.head)

        h = heading_direction / numpy.linalg.norm(heading_direction)
        f = food_direction / numpy.linalg.norm(food_direction)

        fa = math.atan2(h[0] * f[1] - h[1] * f[0], h[0] * f[0] + h[1] * f[1]) / math.pi

        if fa == -1 or fa == 1:
            fa = 1

        self.food_angle = fa

        return self.food_angle

    def get_food_distance(self):
        head = numpy.array(self.snake.head)
        food = numpy.array(self.food.location)

        max_dis = numpy.linalg.norm(numpy.array([0, 0]) - numpy.array([WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1]))
        dis = numpy.linalg.norm(head - food)

        # normalize distance to the range 0 - 1
        self.food_distance = dis / max_dis

        return self.food_distance


# test
def start_agent(code_id=0):
    agent = Agent(code_id, True, False, 30)
    s = agent.get_state()

    while agent.alive:
        # generate move
        move = agent.get_random_move(5)

        # show current state and move
        print(s, move)
        # pre_s = s
        s = agent.next_state(move)

        # Freeze
        # if not agent.alive:
        #     # input('PRESS ENTER TO END')
        #     print(pre_s, move, agent.score)

    return agent.score, agent.step


if __name__ == '__main__':
    num_games = 100

    l_score = []
    l_steps = []

    st = time.time()

    for i in range(num_games):
        score, step = start_agent(i)
        l_score.append(score)
        l_steps.append(step)

    print('_________________________________________________________________')
    print('Time:', time.time() - st)
    print('Total Games:', num_games)
    print('Total Steps:', sum(l_steps))
    print('Avg Steps:', sum(l_steps) / float(len(l_steps)))
    print('Max Score:', max(l_score))
    print('Avg Score:', sum(l_score) / float(len(l_score)))
    print('=================================================================')
