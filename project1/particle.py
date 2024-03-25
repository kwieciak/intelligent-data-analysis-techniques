import random
import math

class Particle:
    def __init__(self, vector_size, inertia, cognitive_constant, social_constant, a, b):
        self.vector_size = vector_size
        self.best_adaptation = float('inf')
        self.velocity = []
        self.inertia = inertia
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.a = a
        self.b = b
        self.vector = self.generate_vector()
        self.best_vector = self.vector

    def generate_vector(self):
        vector = []
        for i in range(self.vector_size):
            vector.append(random.uniform(self.a, self.b))
        return vector

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
