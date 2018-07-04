import numpy as np
import pygame
from utils import *
from constants import *
from random import randint


class Road:

    def __init__(self, size, width):
        x = np.asarray(list(range(20, WIN_SIZE_X - 20, 31)))
        xr = np.asarray(list(range(20, WIN_SIZE_X - 20, 31)))
        y = 30 * np.cos(x)
        # for i, _ in enumerate(xr):
        #     xr[i] += randint(0, 17)
        self.left_wall = list(zip(x, y + width / 2 + size[1] / 2))
        self.right_wall = list(zip(xr, y - width / 2 + size[1] / 2))

    def draw(self, surface):
        pygame.draw.lines(surface, (0, 0, 0), False, flip_and_round_list(self.left_wall), 1)
        pygame.draw.lines(surface, (0, 0, 0), False, flip_and_round_list(self.right_wall), 1)
