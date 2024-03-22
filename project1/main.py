import functions
import differential_evolution_algorithm
if __name__ == '__main__':

    DE = differential_evolution_algorithm.DifferentialEvolution(functions.sphere, 50, 1000, 3, -5, 5)
    iterator = 0
    print(DE.optimize())
    #for best_solution in DE.optimize():
    #    iterator += 1
    #    print(f",{iterator}, Best solution: {best_solution}, Best fitness: {DE.func(best_solution)}")
   # print(f"{DE.population}")
