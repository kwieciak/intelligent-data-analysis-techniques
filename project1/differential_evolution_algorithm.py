import random

import numpy as np


class DifferentialEvolution:
    def __init__(self, func, population_size, iterations, vector_size, a, b, cr=0.7, f=0.5):
        self.func = func
        self.population_size = population_size
        self.vector_size = vector_size
        self.a = a
        self.b = b
        self.cr = cr
        self.f = f
        self.iterations = iterations
        self.population = self.generate_population()
        self.best_solution = self.find_the_best()

    def optimize(self):
        best_solutions = []
        for iterator in range(self.iterations):
            for x in range(0, self.population_size):
                v = self.population[x]
                d, e = self.sample(2, x)
                M = self.mutation(d, e)
                K = self.crossover(M, v)
                if self.func(K) < self.func(v):
                    self.population[x] = K
                if self.func(K) < self.func(self.best_solution):
                    self.best_solution = K
            best_solutions.append(self.best_solution)
        ind = np.argmin(best_solutions)
        return best_solutions

    def generate_population(self):
        population = []
        for i in range(self.population_size):
            vector = []
            for j in range(self.vector_size):
                vector.append(random.uniform(self.a, self.b))
            population.append(vector)
        return population

    def find_the_best(self):
        best_solution = self.population[0]
        best_fitness = self.func(best_solution)
        for i in self.population:
            fitness = self.func(i)
            if fitness < best_fitness:
                best_solution = i
                best_fitness = fitness
        return best_solution

    def sample(self, number, exclude_ind):
        population_ind = list(range(self.population_size))
        population_ind.remove(exclude_ind)
        selected_ind = random.sample(population_ind, number)
        return [self.population[i] for i in selected_ind]

    def crossover(self, m, v):
        new = v.copy()
        for i in range(len(v)):
            if random.uniform(0, 1) < self.cr:
                new[i] = m[i]
        return new

    def mutation(self, d, e):
        new = [self.best_solution[i] + self.f * (d[i] - e[i]) for i in range(self.vector_size)]
        new = [min(max(val, self.a), self.b) for val in new]
        return new
