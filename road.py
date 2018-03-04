import numpy as np
import pygame
from utils import *
from constants import *


class Road:

    def __init__(self, size, width):
        x = np.asarray(list(range(20, WIN_SIZE_X - 20, 30)))
        # x = np.array(range(int((size[0] - 40) / 50)))
        y = 15 * np.cos(x)
        self.left_wall = list(zip(x, y + width / 2 + size[1] / 2))
        self.right_wall = list(zip(x, y - width / 2 + size[1] / 2))

    def draw(self, surface):
        pygame.draw.lines(surface, (0, 0, 0), False, flip_and_round_list(self.left_wall), 1)
        pygame.draw.lines(surface, (0, 0, 0), False, flip_and_round_list(self.right_wall), 1)
        # for p in self.left_wall:
        #     pygame.draw.circle(surface, (0, 0, 0), flip_and_round(p), 2)
        # for p in self.right_wall:
        #     pygame.draw.circle(surface, (0, 0, 0), flip_and_round(p), 2)
