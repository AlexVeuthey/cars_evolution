import torch.nn as nn
import torch.nn.functional as F
from random import randint
from constants import *
import numpy as np
# from torchvision.transforms import ToTensor


class Model(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.h1 = nn.Linear(N_INPUT, hidden_size)
        self.h2 = nn.Linear(hidden_size, N_OUTPUT)

    def forward(self, x):
        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)
        return F.sigmoid(x)

    @staticmethod
    def gen_random():
        model = Model(randint(MIN_HIDDEN, MAX_HIDDEN))
        return model


class Driver:

    def __init__(self, model, car):
        self.model = model
        self.car = car

        # self.tTensor = ToTensor()

    def update(self, road):
        left_wall = road.left_wall
        right_wall = road.right_wall
        d_left = self.car.compute_distance_left(left_wall)
        d_right = self.car.compute_distance_right(right_wall)
        d_front = self.car.compute_distance_front(left_wall, right_wall)
        inpt = np.asarray([d_left, d_right, d_front])/MAX_DISTANCE
        # output = self.model.forward(inpt)
        self.car.update()