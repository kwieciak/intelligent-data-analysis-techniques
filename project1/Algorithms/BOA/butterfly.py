import copy
import random


class Butterfly:
    def __init__(self, vector_size, left_border, right_border):
        self.vector_size = vector_size
        self.left_border = left_border
        self.right_border = right_border
        self.best_adaptation = float('inf')
        self.vector = [random.uniform(self.left_border, self.right_border) for _ in range(self.vector_size)]
        self.best_vector = copy.deepcopy(self.vector)
        self.fragrance = None



