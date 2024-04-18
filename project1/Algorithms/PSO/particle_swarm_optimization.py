import copy
import random

from project1.Algorithms.PSO import particle


class PsoDE:
    def __init__(self, particles_num, a, b, func, inertia, cognitive_constant, social_constant, vector_size,
                 iterations, cr=0.7, f=0.5):
        self.a = a
        self.b = b
        self.func = func
        self.inertia = inertia
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.vector_size = vector_size
        self.iterations = iterations
        self.population = self.generate_population(particles_num)
        self.best_adaptation = float('inf')
        self.best_particle = None
        self.best_values_over_time = []
        self.cr = cr
        self.f = f

    def optimize(self):
        for i in range(self.iterations):
            self.update_adaptation()
            for j in range(len(self.population)):
                self.population[j].update_velocity(self.best_particle)
                self.population[j].update_position()
                d, e = self.sample(2, j)
                M = self.mutation(d.vector, e.vector)
                K = self.crossover(M, self.population[j].vector)
                if self.func(K) < self.func(self.population[j].vector):
                    self.population[j].vector = K
                if self.func(K) < self.best_adaptation:
                    self.best_adaptation = self.func(K)
                    self.best_particle = self.population[j]
            self.best_values_over_time.append(self.best_adaptation)
        return self.best_adaptation, self.best_values_over_time

    def generate_population(self, particles_num):
        particles = []
        for i in range(particles_num):
            new_particle = particle.Particle(self.vector_size, self.inertia, self.cognitive_constant,
                                             self.social_constant, self.a, self.b)
            particles.append(new_particle)
        return particles

    def calculate_adaptation(self, particlee):
        return self.func(particlee.vector)

    def update_adaptation(self):
        for particlee in self.population:
            adaptation = self.calculate_adaptation(particlee)
            if adaptation < particlee.best_adaptation:  # local
                particlee.best_adaptation = adaptation
                particlee.best_vector = copy.deepcopy(particlee.vector)
            if adaptation < self.best_adaptation:  # global
                self.best_adaptation = adaptation
                self.best_particle = copy.deepcopy(particlee)

    def sample(self, number, exclude_ind):
        population_ind = list(range(len(self.population)))
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
        new = [self.best_particle.best_adaptation + self.f * (d[i] - e[i]) for i in range(self.vector_size)]
        new = [min(max(val, self.a), self.b) for val in new]
        return new
