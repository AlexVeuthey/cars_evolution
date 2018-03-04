from constants import *


def flip_and_round(point):
    return [int(point[0]), WIN_SIZE_Y - int(point[1])]

def flip_and_round_list(points):
    new_points = []
    for p in points:
        new_points.append(flip_and_round(p))
    return new_points
