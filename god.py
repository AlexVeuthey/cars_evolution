from driver import Driver, Model
from car import Car
from constants import *
from random import shuffle, randint


class God:

    def __init__(self, generations, start_pos, start_dir):
        """ Initialize the god entity

        :param generations: number of generations
        :param start_pos: starting position for all the cars (no preference given)
        :param start_dir: starting direction for all the cars (no preference given)
        """
        self.population_size = N_DRIVERS
        self.generations = generations
        self.current_gen = 0
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.drivers = []

    def initialize_population(self):
        """ Initializes the population with random models

        :return: None
        """
        self.drivers = []
        for _ in range(self.population_size):
            d = Driver(Model.gen_random(), Car(self.start_pos, self.start_dir))
            self.drivers.append(d)

    def update_once(self, road):
        """ Update all the drivers once, to initialize the distances of all cars

        :param road: the road required for the update
        :return: None
        """
        for d in self.drivers:
            d.update(road)

    def run(self, road):
        """ Target for the thread's main method: update all cars and generate children with genetics
        until max number of generations is reached

        :param road: the road defining the current run
        :return: None
        """
        self.initialize_population()
        while self.current_gen < self.generations:
            print('Starting generation {0}'.format(self.current_gen + 1))
            for d in self.drivers:
                while d.car.alive and not d.car.iterations > MAX_ITERATIONS:
                    d.update(road)
                d.car.alive = False
            self.generate_children()
            self.current_gen += 1

    def draw(self, surface):
        """ Method for displaying the cars

        :param surface: the surface to draw on
        :return: None
        """
        for d in self.drivers:
            d.car.draw(surface)

    def generate_children(self):
        """ Replace the driver's models with a new generation

        :return: None
        """
        self.drivers.sort(key=lambda x: x.car.distance_driven, reverse=True)
        print('Best fitness: {0}'.format(self.drivers[0].car.distance_driven))
        new_drivers = []

        # keep top drivers for next generation, just in case
        for i in range(N_BEST_TO_KEEP):
            driver = Driver(self.drivers[i].model, Car(self.start_pos, self.start_dir))
            new_drivers.append(driver)

        # generate new children from parents
        parents = self.drivers[:N_COMBINED]
        shuffle(parents)
        pairs = []
        # first, pair parents together randomly
        for i, p1 in enumerate(parents):
            if i % 2 == 0:
                pairs.append([p1, parents[i + 1]])
        # then combine each pair together, while also mutating the children a random number of times
        for p in pairs:
            children = Model.combine(p[0], p[1])
            child1 = Driver(children[0], Car(self.start_pos, self.start_dir))
            child2 = Driver(children[1], Car(self.start_pos, self.start_dir))
            for _ in range(randint(0, 20)):
                child1.model.mutate()
            for _ in range(randint(0, 20)):
                child2.model.mutate()
            new_drivers.extend([child1, child2])

        shuffle(new_drivers)
        self.drivers = new_drivers
