import numpy as np

import functions
import differential_evolution_algorithm
from matplotlib import pyplot, pyplot as plt

if __name__ == '__main__':

    DE = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100)
    iterator = 0
    best_solutions = DE.optimize()
    #print(best_solutions[-1])
    #print(functions.sphere(best_solutions[-1]))
    best_solutions_fitness = []
    ind = [i for i in range(1000)]
    for x in best_solutions:
        best_solutions_fitness.append(functions.sphere(x))
        print(iterator)
        print(x)
        print("\n")
        iterator += 1
    ypoints = best_solutions_fitness
    pyplot.plot(ind, ypoints)
    plt.show()
