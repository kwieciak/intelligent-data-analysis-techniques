import copy
import random

import numpy as np

from project1.Algorithms.BOA import butterfly


class BoaAlgorithm:
    def __init__(self, pop_size, vector_size, left_border, right_border, iterations, func, c=0.01, a=0.1, p=0.8):
        self.vector_size = vector_size
        self.iterations = iterations
        self.func = func
        self.left_border = left_border
        self.right_border = right_border
        self.population = self.generate_population(pop_size)
        self.best_adaptation = float('inf')
        self.best_butterfly = None
        self.best_values_over_time = []
        self.c = c
        self.a = a
        self.p = p

    def optimize(self):
        for i in range(self.iterations):
            for j in range(len(self.population)):
                self.population[j].fragrance = self.calculate_f(self.c, self.calculate_adaptation(self.population[j]),
                                                                self.a)
            self.update_adaptation()
            for j in range(len(self.population)):
                r = random.uniform(0, 1)
                if r < self.p:
                    self.population[j].vector = self.add_vectors(self.population[j].vector, self.multiply_vector_by_scalar((self.substraction_vectors(self.multiply_vector_by_scalar(self.best_butterfly.vector, r*r), self.population[j].vector)), self.population[j].fragrance))
                else:
                    rand_but_1, rand_but_2 = self.sample(2, j, len(self.population))
                    self.population[j].vector = self.add_vectors(self.population[j].vector, self.multiply_vector_by_scalar((self.substraction_vectors(self.multiply_vector_by_scalar(rand_but_1.vector, r*r), rand_but_2.vector)), self.population[j].fragrance))
            self.a = 0.1 + (0.2 * (i / self.iterations))
            self.best_values_over_time.append(self.best_adaptation)
        return self.best_adaptation, self.best_values_over_time

    def generate_population(self, pop_size):
        butterflies = []
        for i in range(pop_size):
            new_butterfly = butterfly.Butterfly(self.vector_size, self.left_border, self.right_border)
            butterflies.append(new_butterfly)
        return butterflies

    def calculate_f(self, c, i, a):
        return c * (i ** a)

    def calculate_adaptation(self, butterfly):
        return self.func(butterfly.vector)

    def update_adaptation(self):
        for butterly in self.population:
            adaptation = self.calculate_adaptation(butterly)
            if adaptation < butterly.best_adaptation:  # local
                butterly.best_adaptation = adaptation
                butterly.best_vector = copy.deepcopy(butterly.vector)
            if adaptation < self.best_adaptation:  # global
                self.best_adaptation = adaptation
                self.best_butterfly = copy.deepcopy(butterly)

    def sample(self, number, exclude_ind, pop_size):
        population_ind = list(range(pop_size))
        population_ind.remove(exclude_ind)
        selected_ind = random.sample(population_ind, number)
        return [self.population[i] for i in selected_ind]

    def multiply_vector_by_scalar(self, vector, scalar):
        return [x * scalar for x in vector]

    def substraction_vectors(self, vector1, vector2):
        return [x - y for x, y in zip(vector1, vector2)]

    def add_vectors(self, vector1, vector2):
        return [x + y for x, y in zip(vector1, vector2)]


