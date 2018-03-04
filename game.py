import numpy as np
from god import God
from road import Road
import pygame
from constants import *
from utils import *

size = [WIN_SIZE_X, WIN_SIZE_Y]
screen = pygame.display.set_mode(size)

start_pos = np.array([20, size[1] / 2])
start_dir = 0.0

god = God(1)
god.initialize_population(start_pos, start_dir)

road = Road(size, width=50)

done = False
clock = pygame.time.Clock()

god.update(road)

while not done:
    clock.tick(10)

    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        # handle MOUSEBUTTONUP
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            print(flip_and_round(pos))

    screen.fill((255, 255, 255))

    road.draw(screen)
    # god.update(road)
    god.draw(screen)

    # for loop on generations
    # update god
    # draw

    pygame.display.flip()

pygame.quit()
