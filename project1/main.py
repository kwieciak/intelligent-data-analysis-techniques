from matplotlib import pyplot, pyplot as plt

import differential_evolution_algorithm
import functions
import particle_swarm_optimization

# rosenbrock_bounds = [-2.048, 2.048]
# rastrigin_bounds = [-5.12, 5.12]
# sphere_bounds = [-100, 100]
# griewank_bounds = [-600, 600]

if __name__ == '__main__':
    DE = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100)
    iterator = 0
    best_solutions = DE.optimize()
    #print(best_solutions[-1])
    print("DE:")
    print(functions.sphere(best_solutions[-1]))
    print("\n")
    PSO = particle_swarm_optimization.PsoDE(50, -100, 100, functions.sphere, 0.65, 1.5, 2, 20, 1000)
    best_fit, best_sol, rozwiazania = PSO.optimize()
    #print(best_sol)
    print("PSO_DE:")
    print(best_fit)
    best_solutions_fitness = []
    ind = [i for i in range(1000)]
    for x in rozwiazania:
        best_solutions_fitness.append(x)
        print(iterator)
        print(x)
        print("\n")
        iterator += 1
    ypoints = best_solutions_fitness
    pyplot.plot(ind, ypoints)
    plt.show()
