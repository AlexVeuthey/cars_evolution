import numpy as np
import pygame
from utils import *
from constants import *


class Road:

    def __init__(self, size, width):
        """ Initialize the road

        :param size: the size of the road
        :param width: the width of the screen (to compensate)
        """
        x = np.asarray(list(range(20, WIN_SIZE_X - 20, 31)))
        xr = np.asarray(list(range(20, WIN_SIZE_X - 20, 31)))
        y = 42 * np.cos(x) - 20
        self.left_wall = list(zip(x, y + width / 2 + size[1] / 2))
        self.right_wall = list(zip(xr, y - width / 2 + size[1] / 2))

    def draw(self, surface):
        """ Draws the road on a surface

        :param surface: the surface to draw on
        :return: None
        """
        pygame.draw.lines(surface, (0, 0, 0), False, flip_and_round_list(self.left_wall), 1)
        pygame.draw.lines(surface, (0, 0, 0), False, flip_and_round_list(self.right_wall), 1)
