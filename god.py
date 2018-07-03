from driver import Driver, Model
from car import Car
from constants import *
from random import shuffle


class God:

    def __init__(self, generations, start_pos, start_dir):
        self.population_size = N_DRIVERS
        self.generations = generations
        self.current_gen = 0
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.drivers = []

    def initialize_population(self, road):
        self.drivers = []
        for _ in range(self.population_size):
            driver = Driver(Model.gen_random(), Car(self.start_pos, self.start_dir))
            self.drivers.append(driver)
        self.update(road)

    def run(self, road):
        self.initialize_population(road)
        while self.current_gen < self.generations:
            print('Starting generation {0}'.format(self.current_gen + 1))
            for d in self.drivers:
                while d.car.alive:
                    self.update_driver(road, d)
            self.generate_children()
            self.update(road)
            self.current_gen += 1

    def draw(self, surface):
        for d in self.drivers:
            d.car.draw(surface)

    def update(self, road):
        for d in self.drivers:
            d.update(road)

    def generate_children(self):
        self.drivers.sort(key=lambda x: x.car.distance_driven, reverse=True)
        print('Best fitness: {0}'.format(self.drivers[0].car.distance_driven))
        new_drivers = []

        # keep top drivers for next generation
        new_drivers.extend(self.drivers[:N_BEST_TO_KEEP])

        # generate new children from parents
        for i in range(N_COMBINED):
            new_drivers.append(self.drivers[i])

        # generate new random drivers
        for i in range(N_RANDOM):
            driver = Driver(Model.gen_random(), Car(self.start_pos, self.start_dir))
            new_drivers.append(driver)
        shuffle(new_drivers)
        self.drivers = new_drivers

    @staticmethod
    def update_driver(road, driver):
        driver.update(road)
