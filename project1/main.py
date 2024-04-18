from matplotlib import pyplot as plt
from time import time
from project1.Algorithms.DE import differential_evolution_algorithm
import functions
from project1.Algorithms.PSO import particle_swarm_optimization

# f2_bounds = [-100, 100]
# rastrigin_bounds = [-5.12, 5.12]
# sphere_bounds = [-100, 100]
# griewank_bounds = [-600, 600]

if __name__ == '__main__':
    # time1_start = time()
    # DE1 = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100, cr=0.25,
    #                                                              f=0.5)
    # best_solutions1 = DE1.optimize()
    # time1_end = time()
    # time1 = time1_end - time1_start
    #
    # time2_start = time()
    # DE2 = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100, cr=0.5,
    #                                                              f=0.5)
    # best_solutions2 = DE2.optimize()
    # time2_end = time()
    # time2 = time2_end - time2_start
    #
    # time3_start = time()
    # DE3 = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100, cr=0.75,
    #                                                              f=0.5)
    # best_solutions3 = DE3.optimize()
    # time3_end = time()
    # time3 = time3_end - time3_start

    # print("0.25", best_solutions1[-1], time1)
    # print("0.5", best_solutions2[-1], time2)
    # print("0.75", best_solutions3[-1], time3)

    # time4_start = time()
    # PSODE1 = particle_swarm_optimization.PsoDE(50, -100, 100, functions.sphere, 0.75, 0.5, 0.5, 20, 1000, cr=0.75,
    #                                            f=0.5)
    # best_adapt1, best_solutions4 = PSODE1.optimize()
    # time4_end = time()
    # time4 = time4_end - time4_start
    #
    # time5_start = time()
    # PSODE2 = particle_swarm_optimization.PsoDE(50, -100, 100, functions.sphere, 0.75, 0.5, 1, 20, 1000, cr=0.75, f=0.5)
    # best_adapt2, best_solutions5 = PSODE2.optimize()
    # time5_end = time()
    # time5 = time5_end - time5_start
    #
    # time6_start = time()
    # PSODE3 = particle_swarm_optimization.PsoDE(50, -100, 100, functions.sphere, 0.75, 0.5, 1.5, 20, 1000, cr=0.75,
    #                                            f=0.5)
    # best_adapt3, best_solutions6 = PSODE3.optimize()
    # time6_end = time()
    # time6 = time6_end - time6_start
    #
    # print("25", best_adapt1, time4)
    # print("50", best_adapt2, time5)
    # print("75", best_adapt3, time6)

    time7_start = time()
    PSODE4 = particle_swarm_optimization.PsoDE(50, -100, 100, functions.sphere, 0.75, 0.5, 1, 20, 1000, cr=0.75, f=0.5)
    best_adapt4, best_solutions7 = PSODE4.optimize()
    time7_end = time()
    time7 = time7_end - time7_start

    time8_start = time()
    DE4 = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100, cr=0.75,
                                                                 f=0.5)
    best_solutions8 = DE4.optimize()
    time8_end = time()
    time8 = time8_end - time8_start

    print("PSO_DE Value:", best_adapt4, " Time: ", time7)
    print("DE Value:", best_solutions8[-1], " Time: ", time8)

    def plot(scores, scores2):
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(scores) + 1), scores, label='PSO_DE ', color='green')
        plt.plot(range(1, len(scores2) + 1), scores2, label='DE', color='red')

        plt.xlabel('Iteration')
        plt.ylabel('GRIEWANK FUNCTION VALUE')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    # plot(best_solutions1, best_solutions2, best_solutions3)
    # plot(best_solutions4, best_solutions5, best_solutions6)
    plot(best_solutions7, best_solutions8)