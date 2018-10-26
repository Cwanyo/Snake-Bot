import pygame


class Snake:
    def __init__(self, window_width, window_height, pixel_size, init_x, init_y):
        self.window_width = window_width
        self.window_height = window_height
        self.pixel_size = pixel_size

        self.moves = [[0, -1],  # up
                      [0, 1],  # down
                      [-1, 0],  # right
                      [1, 0]]  # left
        self.moveDirection = self.moves[0]

        self.head_color = (0, 100, 0)
        self.body_color = (0, 200, 0)

        self.head = [init_x, init_y]
        self.body = [[self.head[0], self.head[1] + 1]]
        # for i in range(3, 97):
        #     self.body.append([self.head[0], self.head[1] + i])

    def change_direction(self, direction):
        if direction == self.moves[0] and not self.moveDirection == self.moves[1]:
            self.moveDirection = direction
        elif direction == self.moves[1] and not self.moveDirection == self.moves[0]:
            self.moveDirection = direction
        elif direction == self.moves[2] and not self.moveDirection == self.moves[3]:
            self.moveDirection = direction
        elif direction == self.moves[3] and not self.moveDirection == self.moves[2]:
            self.moveDirection = direction

    def move(self):
        self.body.insert(0, self.head)
        self.body.pop()
        self.head = [self.head[0] + self.moveDirection[0], self.head[1] + self.moveDirection[1]]

    def collision_food(self, food_location):
        if self.head == food_location:
            self.body.insert(-1, self.body[-1])
            return True
        else:
            return False

    def collision_obstacles(self):
        if self.head[0] < 0 or self.head[0] >= self.window_width \
                or self.head[1] < 0 or self.head[1] >= self.window_height:
            return True

        for b in self.body:
            if self.head == b:
                return True

        return False

    def get_length(self):
        return len(self.body) + 1

    def render(self, win):
        for b in self.body:
            pygame.draw.rect(win, self.body_color,
                             (b[0] * self.pixel_size, b[1] * self.pixel_size, self.pixel_size, self.pixel_size))

        pygame.draw.rect(win, self.head_color,
                         (self.head[0] * self.pixel_size, self.head[1] * self.pixel_size, self.pixel_size,
                          self.pixel_size))
