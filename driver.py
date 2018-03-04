import torch.nn as nn
import torch.nn.functional as F
from random import randint

N_INPUT = 3
N_OUTPUT = 2
MIN_HIDDEN = 2
MAX_HIDDEN = 10


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
