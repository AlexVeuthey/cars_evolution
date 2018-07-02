from driver import Driver, Model
from car import Car


class God:

    def __init__(self, pop_size, generations, start_pos, start_dir):
        self.population_size = pop_size
        self.generations = generations
        self.current_gen = 0
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.drivers = []

    def initialize_population(self, road):
        self.drivers = []
        for i in range(self.population_size):
            driver = Driver(Model.gen_random(), Car(self.start_pos, self.start_dir))
            driver.update(road)
            self.drivers.append(driver)

    def run(self, road):
        while self.current_gen < self.generations:
            print('Starting generation {0}'.format(self.current_gen))
            self.initialize_population(road)
            for d in self.drivers:
                while d.car.alive:
                    self.update_driver(road, d)
            self.current_gen += 1

    def draw(self, surface):
        for d in self.drivers:
            d.car.draw(surface)

    def update(self, road):
        for d in self.drivers:
            d.update(road)

    @staticmethod
    def update_driver(road, driver):
        driver.update(road)
