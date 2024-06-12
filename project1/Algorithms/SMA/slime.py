import copy
import random

import numpy as np


class Slime:
    def __init__(self, vector_size, a, b):
        self.vector_size = vector_size
        self.a = a
        self.b = b
        self.vector = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
        self.best_vector = copy.deepcopy(self.vector)
        self.best_adaptation = float('inf')
        self.pattern = copy.deepcopy(self.vector)
        self.w = 0

    def update_position(self, global_best, z, func, s1, s2, vc, a):
        p = np.tanh(abs(func(self.vector) - func(global_best.vector)))
        r = random.uniform(0, 1)
        if random.uniform(0, 1) < z:
            self.vector = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
        elif r < p:
            step1 = self.multiply_vector_by_scalar(s1.vector, self.w)
            step2 = self.subtraction_vectors(step1, s2.vector)
            step3 = self.multiply_vector_by_scalar(step2, random.uniform(-a, a))
            step4 = self.add_vectors(global_best.vector, step3)
            self.vector = step4
        else:
            self.vector = self.multiply_vector_by_scalar(self.vector, random.uniform(-vc,vc))
        self.vector = np.clip(self.vector, self.a, self.b)
        self.best_adaptation = func(self.vector)
        if self.best_adaptation < func(self.best_vector):
            self.best_vector = copy.deepcopy(self.vector)
            self.pattern = copy.deepcopy(self.vector)


    def multiply_vector_by_scalar(self, vector, scalar):
        return [x * scalar for x in vector]

    def subtraction_vectors(self, vector1, vector2):
        return [x - y for x, y in zip(vector1, vector2)]

    def add_vectors(self, vector1, vector2):
        return [x + y for x, y in zip(vector1, vector2)]



