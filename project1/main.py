from matplotlib import pyplot as plt
from time import time

from project1.Algorithms.BAT import bat_algorithm
from project1.Algorithms.BOA import boa_algorithm
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

    # time7_start = time()
    # PSODE4 = particle_swarm_optimization.PsoDE(50, -100, 100, functions.sphere, 0.75, 0.5, 1, 20, 1000, cr=0.75, f=0.5)
    # best_adapt4, best_solutions7 = PSODE4.optimize()
    # time7_end = time()
    # time7 = time7_end - time7_start
    #
    # time8_start = time()
    # DE4 = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 20, -100, 100, cr=0.75,
    #                                                              f=0.5)
    # best_solutions8 = DE4.optimize()
    # time8_end = time()
    # time8 = time8_end - time8_start

    # time_start = time()
    # BAT = bat_algorithm.BatAlgorithm(50, 30, -100, 100, 1000, 0, 10, functions.sphere)
    # best_adapt, best_solutions = BAT.optimize()
    # time_end = time()
    #
    # print("BAT Value:", best_adapt, " Time: ", time_end - time_start)
    # plt.plot(range(1, len(best_solutions) + 1), best_solutions, label='BAT', color='blue')
    # plt.show()

    # time_start = time()
    # BOA = boa_algorithm.BoaAlgorithm(100, 20, -100, 100, 1000, functions.sphere)
    # best_adapt, best_solutions = BOA.optimize_with_levy_flight()
    # time_end = time()
    #
    # print("BOA Value:", best_adapt, " Time: ", time_end - time_start)
    # plt.plot(range(1, len(best_solutions) + 1), best_solutions, label='BOA', color='blue')
    # plt.show()
    #print("DE Value:", best_solutions8[-1], " Time: ", time8)

    time_start1 = time()
    BAT1 = bat_algorithm.BatAlgorithm(25, 30, -100, 100, 1000, 0, 10, functions.griewank_function)
    best_adapt1, best_solutions1 = BAT1.optimize()
    time_end1 = time()

    # time_start2 = time()
    # BAT2 = bat_algorithm.BatAlgorithm(25, 30, -100, 100, 1000, 0, 10, functions.sphere)
    # best_adapt2, best_solutions2 = BAT2.optimize()
    # time_end2 = time()
    #
    # time_start3 = time()
    # BAT3 = bat_algorithm.BatAlgorithm(25, 30, -100, 100, 1000, 0, 20, functions.sphere)
    # best_adapt3, best_solutions3 = BAT3.optimize()
    # time_end3 = time()
    # print("frequency = [0,5]  f(x)=", best_adapt1, " Time: ", time_end1 - time_start1)
    # print("frequency = [0,10]  f(x)=", best_adapt2, " Time: ", time_end2 - time_start2)
    # print("frequency = [0,20]  f(x)=", best_adapt3, " Time: ", time_end3 - time_start3)

    time_start2 = time()
    boa1 = boa_algorithm.BoaAlgorithm(75, 30, -100, 100, 1000, functions.griewank_function)
    best_adapt2, best_solutions2 = boa1.optimize_with_levy_flight()
    time_end2 = time()

    time_start3 = time()
    boa1 = boa_algorithm.BoaAlgorithm(75, 30, -100, 100, 1000, functions.griewank_function)
    best_adapt3, best_solutions3 = boa1.optimize()
    time_end3 = time()

    # time_start2 = time()
    # boa2 = boa_algorithm.BoaAlgorithm(75, 30, -100, 100, 1000, functions.sphere, c=0.1)
    # best_adapt2, best_solutions2 = boa2.optimize_with_levy_flight()
    # time_end2 = time()
    #
    # time_start3 = time()
    # boa3 = boa_algorithm.BoaAlgorithm(75, 30, -100, 100, 1000, functions.sphere, c=1)
    # best_adapt3, best_solutions3 = boa3.optimize_with_levy_flight()
    # time_end3 = time()
    # print("sensor modality = 0.01   f(x)=", best_adapt1, " Time: ", time_end1 - time_start1)
    # print("sensor modality = 0.1  f(x)=", best_adapt2, " Time: ", time_end2 - time_start2)
    # print("sensor modality = 1  f(x)=", best_adapt3, " Time: ", time_end3 - time_start3)

    print("BAT Value:", best_adapt1, " Time: ", time_end1 - time_start1)
    print("BOA Value with Levy Flight:", best_adapt2, " Time: ", time_end2 - time_start2)
    print("BOA Value without Levy Flight:", best_adapt3, " Time: ", time_end3 - time_start3)

    def plot(scores, scores2, scores3):
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(scores) + 1), scores, label='BAT  ', color='green')
        plt.plot(range(1, len(scores2) + 1), scores2, label='BOA with Levy Flight ', color='red')
        plt.plot(range(1, len(scores3) + 1), scores3, label='BOA without Levy Flight ', color='blue')

        plt.xlabel('Iteration')
        plt.ylabel('GRIEWANK FUNCTION VALUE')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    plot(best_solutions1, best_solutions2, best_solutions3)
    # # plot(best_solutions1, best_solutions2, best_solutions3)
    # # plot(best_solutions4, best_solutions5, best_solutions6)
    #plot(best_solutions7, best_solutions8)