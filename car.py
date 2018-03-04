from math import cos, sin
import numpy as np

FOV = 30.0
MAX_DISTANCE = 15.0
LENGTH = 5.0
WIDTH = 2.0


class Car:

    def __init__(self, position, direction):
        # 2d vector (tuple)
        self.position = position
        self.length = LENGTH
        self.width = WIDTH
        # angle in degrees, converted to radians, 0° = right, 90° = upwards
        self.direction = direction
        # angle in degrees,
        self.fov = FOV
        self.max_distance = MAX_DISTANCE

    def rotation_matrix(self):
        d = np.radians(self.direction)
        c = cos(d)
        s = sin(d)
        return np.asarray([[c, -s], [s, c]])

    @staticmethod
    def mult(rot, v):
        return [rot[0, 0] * v[0] + rot[0, 1] * v[1], rot[1, 0] * v[0] + rot[1, 1] * v[1]]

    @staticmethod
    def distance_to_line(p1, v1, p2, v2):
        if np.isclose(v1[0], 0) and np.isclose(v2[1], 0):
            x = p1[0]
            y = p2[1]
        elif np.isclose(v2[0], 0) and np.isclose(v1[1], 0):
            x = p2[0]
            y = p1[1]
        elif np.isclose(v1[0], 0) or np.isclose(v2[0], 0):
            y = ((v1[0] / v1[1]) * p1[1] - (v2[0] / v2[1]) * p2[1] + p2[0] - p1[0]) / (
                    (v1[0] / v1[1]) - (v2[0] / v2[1]))
            x = (v1[0] / v1[1]) * (y - p1[1]) + p1[0]
        else:
            m1 = v1[1] / v1[0]
            m2 = v2[1] / v2[0]
            x = (p2[1] - p1[1] + (m2 * p1[0]) - (m2 * p2[0])) / (m1 - m2)
            y = p1[1] + m1 * (x - p1[0])
        vector_points = np.array([x - p1[0], y - p1[1]])
        return np.linalg.norm(vector_points)

    @staticmethod
    def to_left(pt, base_pt, vector):
        vector_points = np.array([pt[0] - base_pt[0], pt[1] - base_pt[1]])
        vector_points = vector_points / np.linalg.norm(vector_points)
        return np.cross(vector, vector_points) > 0

    @staticmethod
    def to_right(pt, base_pt, vector):
        vector_points = np.array([pt[0] - base_pt[0], pt[1] - base_pt[1]])
        vector_points = vector_points / np.linalg.norm(vector_points)
        return np.cross(vector, vector_points) <= 0

    def compute_distance(self, point, wall, fov, left):
        view_angle = (self.direction + fov) % 360 if left else (self.direction - fov) % 360
        view_angle = np.radians(view_angle)
        vector = [cos(view_angle), sin(view_angle)]
        vector = vector / np.linalg.norm(vector)
        smallest_distance = 2 * self.max_distance
        for i in range(wall[:-1].shape[0]):
            p1, p2 = (wall[i], wall[i + 1]) if left else (wall[i + 1], wall[i])
            if self.to_left(p1, point, vector) and self.to_right(p2, point, vector):
                vector2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                vector2 = vector2 / np.linalg.norm(vector2)
                d = self.distance_to_line(point, vector, p1, vector2)
                smallest_distance = min(smallest_distance, d)
        return min(smallest_distance, self.max_distance)

    def compute_distance_left(self, left_wall):
        fl_point = self.position + self.mult(self.rotation_matrix(), np.array([self.length / 2, self.width / 2]))
        return self.compute_distance(fl_point, left_wall, self.fov, True)

    def compute_distance_right(self, right_wall):
        fr_point = self.position + self.mult(self.rotation_matrix(), np.array([self.length / 2, -self.width / 2]))
        return self.compute_distance(fr_point, right_wall, self.fov, False)

    def compute_distance_front(self, left_wall, right_wall):
        f_point = self.position + self.mult(self.rotation_matrix(), np.array([self.length / 2, 0]))
        d_left = self.compute_distance(f_point, left_wall, 0, True)
        d_right = self.compute_distance(f_point, right_wall, 0, False)
        return min(d_left, d_right, self.max_distance)
