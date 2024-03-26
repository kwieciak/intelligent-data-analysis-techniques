import random
import copy
import numpy as np


class Particle:
    def __init__(self, vector_size, inertia, cognitive_constant, social_constant, a, b):
        self.vector_size = vector_size
        self.best_adaptation = float('inf')
        self.inertia = inertia
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.a = a
        self.b = b
        self.vector = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
        self.best_vector = copy.deepcopy(self.vector)
        self.velocity = np.zeros(vector_size)

    def update_velocity(self, best_particle):
        for i in range(len(self.vector)):
            inertia_value = self.inertia * self.velocity[i]
            social_component = self.social_constant * random.uniform(0, 1) * (best_particle.vector[i] - self.vector[i])
            cognitive_component = self.cognitive_constant * random.uniform(0, 1) * (
                    self.best_vector[i] - self.vector[i])
            self.velocity[i] = inertia_value + social_component + cognitive_component

    def update_position(self):
        for i in range(len(self.vector)):
            self.vector[i] += self.velocity[i]
            self.vector[i] = min(max(self.vector[i], self.a), self.b)
