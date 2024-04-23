import copy
import random

import numpy as np

from project1.Algorithms.BAT import bat


class BatAlgorithm:
    def __init__(self, pop_size, vector_size, a, b, iterations, f_min, f_max, func):
        self.vector_size = vector_size
        self.a = a
        self.b = b
        self.iterations = iterations
        self.f_min = f_min
        self.f_max = f_max
        self.func = func
        self.population = self.generate_population(pop_size)
        self.best_adaptation = float('inf')
        self.best_bat = None
        self.best_values_over_time = []

    def optimize(self):
        for i in range(self.iterations):
            self.update_adaptation()
            for j in range(len(self.population)):
                #self.population[j].update_frequency()
                self.population[j].update_velocity(self.best_bat)
                self.population[j].update_position()
                if random.uniform(0, 1) > self.population[j].pulse_rate:
                    self.population[j].local_search(self.best_bat, self.calculate_mean_pulse_rate(self.population))
                if (self.calculate_adaptation(self.population[j]) < self.best_adaptation) and (random.uniform(0, 1) < self.population[j].loudness):
                    self.population[j].loudness = self.population[j].loudness * random.uniform(0,1)
                    self.population[j].pulse_rate = self.population[j].pulse_rate * (1 - np.exp(-self.population[j].loudness * i))
                if self.calculate_adaptation(self.population[j]) < self.best_adaptation:
                    self.best_bat = copy.deepcopy(self.population[j])
                    self.best_adaptation = self.calculate_adaptation(self.population[j])
            self.best_values_over_time.append(self.best_adaptation)
        return self.best_adaptation, self.best_values_over_time

    def generate_population(self, pop_size):
        bats = []
        for i in range(pop_size):
            new_bat = bat.Bat(self.vector_size, self.a, self.b, self.f_min, self.f_max)
            bats.append(new_bat)
        return bats

    def calculate_adaptation(self, bat):
        return self.func(bat.vector)

    def update_adaptation(self):
        for bat in self.population:
            adaptation = self.calculate_adaptation(bat)
            if adaptation < bat.best_adaptation:  # local
                bat.best_adaptation = adaptation
                bat.best_vector = copy.deepcopy(bat.vector)
            if adaptation < self.best_adaptation:  # global
                self.best_adaptation = adaptation
                self.best_bat = copy.deepcopy(bat)

    def calculate_mean_pulse_rate(self, population):
        pulse_rate = 0
        for bat in population:
            pulse_rate += bat.pulse_rate
        return pulse_rate / len(population)