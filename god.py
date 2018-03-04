from driver import Driver, Model
from car import Car


class God:

    def __init__(self, pop_size):
        self.population_size = pop_size
        self.drivers = []

    def initialize_population(self, position, direction):
        self.drivers = []
        for i in range(self.population_size):
            model = Model.gen_random()
            self.drivers.append(Driver(model, Car(position, direction)))
