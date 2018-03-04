import numpy as np
from god import God
from road import Road
import pygame

start_pos = np.array([200, 150])
start_dir = 0.0

god = God(20)
god.initialize_population(start_pos, start_dir)

size = [1200, 600]
screen = pygame.display.set_mode(size)

road = Road(size, width=50)

done = False
clock = pygame.time.Clock()

while not done:
    clock.tick(10)

    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop

    screen.fill((255, 255, 255))

    road.draw(screen)

    pygame.display.flip()

pygame.quit()
