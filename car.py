from math import cos, sin
import numpy as np
import pygame
from utils import flip_and_round
from constants import *


class Car:

    def __init__(self, position, direction):
        """ Initialize a car with a position and direction

        :param position: the starting position
        :param direction: the starting direction
        """
        # 2d vector (tuple)
        self.position = np.copy(position)
        # direction given in degrees
        # 0deg = right, 90deg = downwards because image coordinates
        self.direction = direction
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        # field of view is given in degrees
        self.fov = FOV
        self.max_distance = MAX_DISTANCE

        # initialize intersection points to top-left corner (displayed is V flipped => bottom-left)
        self.inter_point_left = [0, 0]
        self.inter_point_right = [0, 0]
        self.inter_point_front = [0, 0]

        # initialize distances to maximum distance
        self.distance_left = MAX_DISTANCE
        self.distance_right = MAX_DISTANCE
        self.distance_front = MAX_DISTANCE

        # initialize motion properties
        self.speed = 0.0
        self.acceleration = 0.0
        self.steering = 0.0

        # initialize performance values
        self.distance_driven = 0.0
        self.alive = True
        self.started = False
        self.iterations = 0

    def update(self, acceleration, steering):
        """Updates motion and performance properties of the current car

        :param acceleration: value computed by the network for accelerating (or decelerating)
        :param steering: value computed by the network for steering left or right
        :return: None
        """
        if not self.started:
            self.started = True
        if self.alive:
            # check if too close to a wall (crashed) or going backwards (useless child) or too slow (uninteresting)
            if np.isclose(self.distance_left, 0, atol=TOLERANCE) or \
                    np.isclose(self.distance_right, 0, atol=TOLERANCE) or \
                    np.isclose(self.distance_front, 0, atol=TOLERANCE) or \
                    self.speed < 0 or \
                    self.distance_driven > 0 and self.speed < 0.002:
                self.alive = False
                self.speed = 0.0
            else:
                # increment position, speed etc.
                self.iterations += 1
                self.acceleration = acceleration
                self.steering = steering
                self.speed = self.speed + self.acceleration
                self.speed = self.speed * (1.0 - SPEED_DAMPING)
                self.speed = min(self.speed, MAX_SPEED)
                normalized_speed = abs(self.speed / MAX_SPEED)
                self.direction += 10 * self.steering * normalized_speed
                d = np.radians(self.direction)
                vector = [cos(d), sin(d)]
                vector = vector / np.linalg.norm(vector)
                step = vector * self.speed
                step_size = np.linalg.norm(step)
                self.position += step
                step_size = step_size if self.speed > 0 else -step_size
                self.distance_driven += step_size

    def rotation_matrix(self):
        """ Helper function to compute the current car's rotation matrix

        :return: a 2x2 rotation matrix for the current direction
        """
        # first convert the direction in radians
        d = np.radians(self.direction)
        c = cos(d)
        s = sin(d)
        return np.asarray([[c, -s], [s, c]])

    def get_point_position(self, position):
        """ Computes the position of a point on the car

        :param
            position: the name of the point to select, in ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft', 'Front']
        :return:
            a numpy tuple containing the exact position of the selected point
        """
        if position == 'TopLeft':
            vec = np.asarray([self.length / 2, self.width / 2])
        elif position == 'TopRight':
            vec = np.asarray([self.length / 2, -self.width / 2])
        elif position == 'BottomRight':
            vec = np.asarray([-self.length / 2, -self.width / 2])
        elif position == 'BottomLeft':
            vec = np.asarray([-self.length / 2, self.width / 2])
        elif position == 'Front':
            vec = np.asarray([self.length / 2, 0])
        else:
            vec = np.asarray([0, 0])
        return self.position + self.mult(self.rotation_matrix(), vec)

    def draw(self, surface):
        """ Draws the elements of a car on the screen

        :param surface: the screen element to draw on
        :return: None
        """
        condition = self.alive and self.started
        fl = flip_and_round(self.get_point_position('TopLeft'))
        fr = flip_and_round(self.get_point_position('TopRight'))
        br = flip_and_round(self.get_point_position('BottomRight'))
        bl = flip_and_round(self.get_point_position('BottomLeft'))
        # car main block
        color = C_ACTIVE_BODY if condition else C_DEAD_BODY
        pygame.draw.polygon(surface, color, [fl, fr, br, bl])
        # car lights for readability
        color = C_ACTIVE_LIGHTS_F if condition else C_DEAD_LIGHTS_F
        pygame.draw.circle(surface, color, fl, 2)
        pygame.draw.circle(surface, color, fr, 2)
        color = C_ACTIVE_LIGHTS_R if condition else C_DEAD_LIGHTS_R
        pygame.draw.circle(surface, color, br, 2)
        pygame.draw.circle(surface, color, bl, 2)

    @staticmethod
    def mult(rot, v):
        """ Helper method for matrix-vector multiplication (for rotating a point)

        :param rot: the 2x2 rotation matrix
        :param v: the 2x1 position vector to rotate
        :return: a 2x1 rotated vector
        """
        return [rot[0, 0] * v[0] + rot[0, 1] * v[1], rot[1, 0] * v[0] + rot[1, 1] * v[1]]

    @staticmethod
    def distance_to_line(car_point, car_vector, wall_point, wall_vector):
        """ Computes the distance from a point to a wall segment (not necessarily orthogonal distance)

        :param car_point: the point of interest
        :param car_vector: the direction in which to compute the distance
        :param wall_point: a point defining a wall segment start
        :param wall_vector: a vector defining a wall segment's direction
        :return: the computed distance
        """
        # orthogonal case (vertical car vector, horizontal wall vector)
        if np.isclose(car_vector[0], 0) and np.isclose(wall_vector[1], 0):
            x = car_point[0]
            y = wall_point[1]
        # orthogonal case (horizontal car vector, vertical wall vector)
        elif np.isclose(wall_vector[0], 0) and np.isclose(car_vector[1], 0):
            x = wall_point[0]
            y = car_point[1]
        # parallel case (horizontal car and wall vectors)
        elif np.isclose(car_vector[0], 0) or np.isclose(wall_vector[0], 0):
            y = ((car_vector[0] / car_vector[1]) * car_point[1] - (wall_vector[0] / wall_vector[1]) * wall_point[1] +
                 wall_point[0] - car_point[0]) / (
                        (car_vector[0] / car_vector[1]) - (wall_vector[0] / wall_vector[1]))
            x = (car_vector[0] / car_vector[1]) * (y - car_point[1]) + car_point[0]
        # every other case does not require imply division by 0
        else:
            car_slope = car_vector[1] / car_vector[0]
            wall_slope = wall_vector[1] / wall_vector[0]
            x = (wall_point[1] - car_point[1] + (car_slope * car_point[0]) - (wall_slope * wall_point[0])) / (
                    car_slope - wall_slope)
            y = car_point[1] + car_slope * (x - car_point[0])
        vector_points = np.array([x - car_point[0], y - car_point[1]])
        return np.linalg.norm(vector_points), [x, y], vector_points

    @staticmethod
    def to_left(pt, base_pt, vector):
        """ Checks whether a point is to the left of a line (defined by point + vector)

        :param pt: the point of which we want the relative position
        :param base_pt: the point defining the line
        :param vector: the direction vector defining the line
        :return: True if the point is to the left, False if aligned or to the right
        """
        vector_points = np.array([pt[0] - base_pt[0], pt[1] - base_pt[1]])
        vector_points = vector_points / np.linalg.norm(vector_points)
        c = np.cross(vector, vector_points)
        return c > 0

    @staticmethod
    def to_right(pt, base_pt, vector):
        """ Checks whether a point is to the right of a line (defined by point + vector)

        :param pt: the point of which we want the relative position
        :param base_pt: the point defining the line
        :param vector: the direction vector defining the line
        :return: True if the point is to the right or aligned, False if to the left
        """
        vector_points = np.array([pt[0] - base_pt[0], pt[1] - base_pt[1]])
        vector_points = vector_points / np.linalg.norm(vector_points)
        c = np.cross(vector, vector_points)
        return c <= 0

    def compute_distance(self, point, wall, fov, left):
        """ Computes the distance (left or right) from a point to a complete wall
        This function is the most inefficient part of the whole program, a new strategy should be implemented.

        :param point: the point of interest
        :param wall: the complete wall segment
        :param fov: the field of view of the car
        :param left: boolean defining the behaviour of the function
        (True is for checking left distance, False right distance)
        :return: the obtained distance, as well as the intersection point
        """
        view_angle = (self.direction + fov) % 360 if left else (self.direction - fov) % 360
        view_angle = np.radians(view_angle)
        vector = [cos(view_angle), sin(view_angle)]
        vector = vector / np.linalg.norm(vector)
        smallest_distance = self.max_distance
        best_inter_point = point + self.max_distance * vector
        # check all wall segments and keep the shortest distance
        for i in range(len(wall[:-1])):
            p1, p2 = (wall[i], wall[i + 1]) if left else (wall[i + 1], wall[i])
            if self.to_left(p1, point, vector) and self.to_right(p2, point, vector):
                vector2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                vector2 = vector2 / np.linalg.norm(vector2)
                d, inter_point, vector_points = self.distance_to_line(point, vector, p1, vector2)
                if d < smallest_distance and np.dot(vector, vector_points) >= 0:
                    smallest_distance = d
                    best_inter_point = inter_point
        return min(smallest_distance, self.max_distance), best_inter_point

    def compute_distance_left(self, left_wall):
        """ Wrapper for left wall distance computing

        :param left_wall: the left wall segments, as a list
        :return: the distance obtained
        """
        fl_point = self.position + self.mult(self.rotation_matrix(), np.array([self.length / 2, self.width / 2]))
        d, point = self.compute_distance(fl_point, left_wall, self.fov, True)
        self.inter_point_left = point
        self.distance_left = d
        return d

    def compute_distance_right(self, right_wall):
        """ Wrapper for right wall distance computing

        :param right_wall: the right wall segments, as a list
        :return: the distance obtained
        """
        fr_point = self.position + self.mult(self.rotation_matrix(), np.array([self.length / 2, -self.width / 2]))
        d, point = self.compute_distance(fr_point, right_wall, self.fov, False)
        self.inter_point_right = point
        self.distance_right = d
        return d

    def compute_distance_front(self, left_wall, right_wall):
        """ Computes the distance in front of the car, which requires both left and right walls

        :param left_wall: the left wall segments, as a list
        :param right_wall: the right wall segments, as a list
        :return: the shortest distance between left and right wall distances
        """
        f_point = self.position + self.mult(self.rotation_matrix(), np.array([self.length / 2, 0]))
        # compute both distances first
        d_left, best_point_left = self.compute_distance(f_point, left_wall, 0, True)
        d_right, best_point_right = self.compute_distance(f_point, right_wall, 0, False)
        # then select which to use depending on the case
        if self.max_distance < d_left and self.max_distance < d_right:
            self.inter_point_front = best_point_left
            self.distance_front = self.max_distance
            return self.max_distance
        elif d_left < d_right:
            self.inter_point_front = best_point_left
            self.distance_front = d_left
            return d_left
        else:
            self.inter_point_front = best_point_right
            self.distance_front = d_right
            return d_right
