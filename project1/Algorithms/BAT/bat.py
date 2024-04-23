import copy
import random


class Bat:
    def __init__(self, vector_size, a, b, f_min, f_max):
        self.vector_size = vector_size
        self.a = a
        self.b = b
        self.f_min = f_min
        self.f_max = f_max
        self.best_adaptation = float('inf')
        self.vector = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
        self.best_vector = copy.deepcopy(self.vector)
        self.velocity = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
        self.frequency = f_min + (f_max - f_min) * random.uniform(0, 1)
        self.pulse_rate = random.uniform(0, 1)
        self.loudness = random.uniform(1, 2)

    def update_velocity(self, best_bat):
        for i in range(len(self.vector)):
            self.velocity[i] = self.velocity[i] + (self.vector[i] - best_bat.vector[i]) * self.frequency
            self.velocity[i] = min(max(self.velocity[i], self.a), self.b)

    def update_position(self):
        for i in range(len(self.vector)):
            self.vector[i] = self.vector[i] + self.velocity[i]
            self.vector[i] = min(max(self.vector[i], self.a), self.b)

    def local_search(self, best_bat, mean_pulse_rate):
        for i in range(len(self.vector)):
            self.vector[i] = best_bat.vector[i] + random.uniform(-1, 1) * mean_pulse_rate
            self.vector[i] = min(max(self.vector[i], self.a), self.b)
