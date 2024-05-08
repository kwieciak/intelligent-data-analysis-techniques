import numpy as np


class BOA:
    def __init__(self, n, dim, max_iter, function, min_bounds, max_bounds):
        self.n = n
        self.dim = dim
        self.max_iter = max_iter
        self.function = function
        self.population = np.random.uniform(min_bounds, max_bounds, (n, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.c = 0.1
        self.a = 0.1
        self.t = 0

    def update_a(self):
        self.a = 0.1 + 0.2 * self.t / self.max_iter

    def calculate_intensity(self):
        return self.c * (self.best_fitness ** self.a)

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * (2 * ((beta - 1) / 2)))) * (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def move_towards_global_best(self):
        r = np.random.rand(self.n, self.dim)
        levy = self.levy_flight()
        for i in range(self.n):
            self.population[i] += levy * (self.best_solution - self.population[i]) * self.calculate_intensity()

    def move_towards_random(self):
        r = np.random.rand(self.n, self.dim)
        levy = self.levy_flight()
        for i in range(self.n):
            j, k = np.random.choice(range(self.n), 2, replace=False)
            self.population[i] +=  levy* (self.population[j] - self.population[k]) * self.calculate_intensity()