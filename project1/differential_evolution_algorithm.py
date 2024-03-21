import copy
import random
import numpy as np


class DifferentialEvolution:
    def __init__(self, func, population_size, iterations, vector_size, a, b, cr=0.5, f=0.5):
        self.func = func
        self.population_size = population_size
        self.vector_size = vector_size
        self.a = a
        self.b = b
        self.cr = cr
        self.f = f
        self.iterations = iterations
        self.population = self.generate_population()
        self.v = self.find_the_best()

    def optimize(self):
        best_solution = self.v
        best_fitness = self.func(best_solution)
        for iterator in range(self.iterations):
            for x in range(0, self.population_size):
                d, e = self.sample(2)
                M = self.mutation(d, e)
                K = self.crossover(M)
                if self.func(K) <= best_fitness:
                    best_solution = K
                    best_fitness = self.func(K)
        return best_solution

    def generate_population(self):
        population = []
        for i in range(self.population_size):
            vector = []
            for j in range(self.vector_size):
                vector.append(random.uniform(self.a, self.b))
            population.append(vector)
        return population

    def find_the_best(self):
        best = self.population[0]
        for i in range(len(self.population) - 1):
            if self.func(self.population[i]) >= self.func(self.population[i + 1]):
                best = self.population[i + 1]
        return best

    def sample(self, number):
        return random.sample(self.population, number)

    def crossover(self, m):
        new = copy.deepcopy(self.v)
        for i in range(len(self.v)):
            if random.uniform(0, 1) < self.cr:
                new[i] = m[i]
        return new

    def mutation(self, d, e):
        new = [self.v[i] + self.f * (d[i] - e[i]) for i in range(self.vector_size)]
        return new
