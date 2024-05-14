import copy
import math
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
        self.c = c #senson modality
        self.a = a #power exponent
        self.p = p #probability

    def optimize(self):
        for i in range(self.iterations):
            for j in range(len(self.population)):
                self.population[j].fragrance = self.calculate_f(self.c, self.calculate_adaptation(self.population[j]),
                                                                self.a)
            self.update_adaptation()
            for j in range(len(self.population)):
                r = random.uniform(0, 1)
                if r < self.p:
                    self.population[j].vector = self.add_vectors(self.population[j].vector, self.multiply_vector_by_scalar(
                        (self.substraction_vectors(self.multiply_vector_by_scalar(self.best_butterfly.vector, r*r),
                                                   self.population[j].vector)), self.population[j].fragrance))
                else:
                    rand_but_1, rand_but_2 = self.sample(2, j, len(self.population))
                    self.population[j].vector = self.add_vectors(self.population[j].vector, self.multiply_vector_by_scalar(
                        (self.substraction_vectors(self.multiply_vector_by_scalar(rand_but_1.vector, r*r), rand_but_2.vector)),
                        self.population[j].fragrance))
            self.a = 0.1 + (0.2 * (i / self.iterations))
            self.best_values_over_time.append(self.best_adaptation)
        return self.best_adaptation, self.best_values_over_time

    def optimize_with_levy_flight(self):
        for i in range(self.iterations):
            for j in range(len(self.population)):
                self.population[j].fragrance = self.calculate_f(self.c, self.calculate_adaptation(self.population[j]),
                                                                self.a) / self.best_adaptation + 2.2204e-16
            self.update_adaptation()
            for j in range(len(self.population)):
                r = random.uniform(0, 1)
                if r < self.p:
                    step = self.levy_flight()
                    local_search_vector = self.multiply_vectors(
                        self.substraction_vectors(self.best_butterfly.vector, self.population[j].vector), step)
                    self.population[j].vector = self.add_vectors(self.population[j].vector,
                                                                 self.multiply_vector_by_scalar(local_search_vector,
                                                                                                self.population[j].fragrance))
                else:
                    rand_but_1, rand_but_2 = self.sample(2, j, len(self.population))
                    M1 = self.mutation(rand_but_1.vector, rand_but_2.vector)
                    K1 = self.crossover(M1, self.population[j].vector)
                    rand_but_3, rand_but_4 = self.sample(2, j, len(self.population))
                    M2 = self.mutation(rand_but_3.vector, rand_but_4.vector)
                    K2 = self.crossover(M2, self.population[j].vector)
                    step = self.levy_flight()
                    global_search_vector = self.multiply_vectors(
                        self.substraction_vectors(K1, K2), step)
                    self.population[j].vector = self.add_vectors(self.population[j].vector,
                                                                 self.multiply_vector_by_scalar(global_search_vector,
                                                                                                self.population[j].fragrance))
            self.a = 0.1 + (0.2 * (i / self.iterations))
            self.c = 1 - 0.6 * math.sqrt(i / self.iterations)
            self.best_values_over_time.append(self.best_adaptation)
        return self.best_adaptation, self.best_values_over_time

    def generate_population(self, pop_size):
        butterflies = []
        for i in range(pop_size):
            new_butterfly = butterfly.Butterfly(self.vector_size, self.left_border, self.right_border)
            butterflies.append(new_butterfly)
        return butterflies

    def calculate_f(self, c, i, a): #i = intensity, f = fragrance
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

    def multiply_vectors(self, vector1, vector2):
        return [x * y for x, y in zip(vector1, vector2)]

    def levy_flight(self):
        lambda_sign = 1.5
        sigma = (np.math.gamma(1 + lambda_sign) * np.sin(np.pi * lambda_sign / 2) / (
                np.math.gamma((1 + lambda_sign) / 2) * lambda_sign ** (2 * ((lambda_sign - 1) / 2)))) ** (
                        1 / lambda_sign)
        u = np.random.normal(0, sigma, self.vector_size)
        v = np.random.standard_normal(self.vector_size)
        epsilon = 1e-10
        step = u / (np.abs(v) ** (1 / lambda_sign) + epsilon)
        return 0.1 * step

    def crossover(self, m, v):
        new = v.copy()
        for i in range(len(v)):
            if random.uniform(0, 1) < 0.7:
                new[i] = m[i]
        return new

    def mutation(self, d, e):
        new = [self.best_butterfly.vector[i] + 0.5 * (d[i] - e[i]) for i in range(self.vector_size)]
        new = [min(max(val, self.left_border), self.right_border) for val in new]
        return new
