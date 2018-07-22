import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
from random import randint, choice


class Model(nn.Module):

    def __init__(self, hidden_size):
        """ Initialize a model given the hidden size

        :param hidden_size: the number of hidden neurons (only 1 hidden layer for now)
        """
        super().__init__()
        self.h1 = nn.Linear(N_INPUT, hidden_size)
        self.h2 = nn.Linear(hidden_size, N_OUTPUT)

    def forward(self, x):
        """ Overrides the nn.Module forward function, for custom models

        :param x: the input value
        :return: the output of the network, of size (N_OUTPUT, 1)
        """
        x = self.h1(x)
        x = F.sigmoid(x)
        x = self.h2(x)
        return F.sigmoid(x)

    def mutate(self):
        """ Mutates the current model randomly

        :return: None
        """
        # select a parameter to mutate
        mutation = choice(list(self.state_dict().keys()))
        matrix = self.state_dict()[mutation]
        dim = list(matrix.shape)
        # if bias "matrix" was selected, we only have 1 dimension (vector)
        if len(dim) == 1:
            index = randint(0, dim[0] - 1)
            matrix[index] = matrix[randint(0, dim[0] - 1)]
        # otherwise, select a random position which gets attributed a randomly selected existing weight
        else:
            index1 = randint(0, dim[0] - 1)
            index2 = randint(0, dim[1] - 1)
            matrix[index1, index2] = matrix[randint(0, dim[0] - 1), randint(0, dim[1] - 1)]
        self.state_dict()[mutation].copy_(matrix)

    @staticmethod
    def gen_random():
        """ Generates a random model

        :return: the randomly generated model
        """
        model = Model(N_HIDDEN)
        return model

    @staticmethod
    def combine(parent1, parent2):
        """ Breed two networks together, using a crossover point for each weight/bias matrices pairs

        :param parent1: first parent to breed
        :param parent2: second parent to breed
        :return: two children defined by a combination of the first and second parents' weight/bias matrices
        """
        # first layer
        weights11 = parent1.model.h1.weight
        weights12 = parent2.model.h1.weight
        bias11 = parent1.model.h1.bias
        bias12 = parent2.model.h1.bias
        crossover1 = randint(0, N_HIDDEN - 1)

        child_weights11 = torch.cat([weights11[:crossover1], weights12[crossover1:]], 0)
        child_weights12 = torch.cat([weights12[:crossover1], weights11[crossover1:]], 0)
        child_bias11 = torch.cat([bias11[:crossover1], bias12[crossover1:]], 0)
        child_bias12 = torch.cat([bias12[:crossover1], bias11[crossover1:]], 0)

        # second layer
        weights21 = parent1.model.h2.weight
        weights22 = parent2.model.h2.weight
        bias21 = parent1.model.h2.bias
        bias22 = parent2.model.h2.bias
        crossover2 = randint(0, N_OUTPUT - 1)

        child_weights21 = torch.cat([weights21[:crossover2], weights22[crossover2:]], 0)
        child_weights22 = torch.cat([weights22[:crossover2], weights21[crossover2:]], 0)
        child_bias21 = torch.cat([bias21[:crossover2], bias22[crossover2:]], 0)
        child_bias22 = torch.cat([bias22[:crossover2], bias21[crossover2:]], 0)

        # create the models and assign weights
        child1 = Model(N_HIDDEN)
        child2 = Model(N_HIDDEN)

        child1.state_dict()['h1.weight'].copy_(child_weights11)
        child1.state_dict()['h1.bias'].copy_(child_bias11)
        child1.state_dict()['h2.weight'].copy_(child_weights21)
        child1.state_dict()['h2.bias'].copy_(child_bias21)

        child2.state_dict()['h1.weight'].copy_(child_weights12)
        child2.state_dict()['h1.bias'].copy_(child_bias12)
        child2.state_dict()['h2.weight'].copy_(child_weights22)
        child2.state_dict()['h2.bias'].copy_(child_bias22)

        return child1, child2


class Driver:

    def __init__(self, model, car):
        """ Initialize a driver with a brain (model) and a car to pilot

        :param model: the model to use
        :param car: the car to pilot
        """
        self.model = model
        self.car = car

    def update(self, road):
        """ Perform one update as the driver (using the model to predict the reaction to the car's distance sensors)

        :param road: the road, which is visible to the driver
        :return: None
        """
        left_wall = road.left_wall
        right_wall = road.right_wall
        d_left = self.car.compute_distance_left(left_wall)
        d_right = self.car.compute_distance_right(right_wall)
        d_front = self.car.compute_distance_front(left_wall, right_wall)
        inpt = torch.Tensor([d_left, d_right, d_front])
        output = self.model.forward(inpt)
        # for real steering and acceleration values, we need to shift by -0.5,
        # as we have a sigmoid output (between 0 and 1)
        self.car.update(output[0].data - 0.5, output[1].data - 0.5)
