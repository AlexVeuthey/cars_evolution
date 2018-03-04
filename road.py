import numpy as np
import pygame


class Road:

    def __init__(self, size, width):
        x = np.array(range(int((size[0]-40)/10)))
        y = 5 * np.cos(x / 2)
        self.left_wall = list(zip(10 * x + 20, y + size[1]/2))
        self.right_wall = list(zip(10 * x + 20, y - width + size[1]/2))

    def draw(self, surface):
        pygame.draw.lines(surface, (0, 0, 0), False, self.left_wall, 1)
        pygame.draw.lines(surface, (50, 50, 50), False, self.right_wall, 1)
