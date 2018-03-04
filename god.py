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

    def draw(self, surface):
        for d in self.drivers:
            d.car.draw(surface)

    def update(self, road):
        for d in self.drivers:
            d.update(road)

    def run(self, road):
        while 1:
            self.update(road)
