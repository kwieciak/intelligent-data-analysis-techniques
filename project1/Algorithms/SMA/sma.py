import random
import copy
import math
import numpy as np

from project1.Algorithms.SMA.slime import Slime


class SlimeMouldSwarm:
    def __init__(self, population_size, a, b, func, vector_size, iterations, patience=10, communication_interval=10):
        self.a = a
        self.b = b
        self.func = func
        self.vector_size = vector_size
        self.iterations = iterations
        self.population = self.generate_population(population_size)
        self.best_adaptation = float('inf')
        self.best_individual = None
        self.best_values_over_time = []
        self.z = random.uniform(0, 1)
        self.vc = 1
        self.patience = patience
        self.stagnation_counter = 0
        self.communication_interval = communication_interval

    def optimize(self, global_best_vector=None):
        for i in range(self.iterations):
            prev_best_adaptation = self.best_adaptation
            self.update_adaptation()
            best_fitness = self.best_adaptation
            worst_fitness = max([self.func(ind.vector) for ind in self.population])

            for k in range(len(self.population)):
                if k < len(self.population) / 2:
                    self.population[k].w = 1 + random.uniform(0, 1) * math.log(
                        (best_fitness - self.func(self.population[k].vector)) / (
                                best_fitness - worst_fitness + 0.00000001) + 1)
                else:
                    self.population[k].w = 1 - random.uniform(0, 1) * math.log(
                        (best_fitness - self.func(self.population[k].vector)) / (
                                best_fitness - worst_fitness + 0.00000001) + 1)

            a = np.arctanh(-(i / self.iterations) + 1)
            self.vc = 1 - (i / self.iterations)

            for k in range(len(self.population)):
                s1, s2 = self.sample(2, k)
                self.population[k].update_position(self.best_individual, self.z, self.func, s1, s2, self.vc, a)
                o = Slime(self.vector_size, self.a, self.b)

                for j in range(len(self.population[k].vector)):
                    d, ind_d = self.sample1(1, k)  # randomly selected particle
                    if self.func(self.population[k].best_vector) < self.func(
                            self.population[ind_d[0]].best_vector):  # crossover
                        r = random.uniform(0, 1)
                        o.vector[j] = r * self.population[k].best_vector[j] + (
                                (1 - r) * self.best_individual.best_vector[j])
                    else:
                        o.vector[j] = self.population[ind_d[0]].vector[j]

                for j in range(len(self.population[k].vector)):  # mutation
                    if random.uniform(0, 1) < 0.6:
                        o.vector[j] = random.uniform(self.a, self.b)

                o_adaptation = self.func(o.vector)
                if o_adaptation < self.func(self.population[k].vector):
                    self.population[k] = o
                if o_adaptation < self.best_adaptation:
                    self.best_adaptation = o_adaptation
                    self.best_individual = o

            if self.best_adaptation == prev_best_adaptation:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter >= self.patience:
                self.tournament_selection()
                self.stagnation_counter = 0

            if (i + 1) % self.communication_interval == 0:
                self.communicate_best_solution()
            self.best_values_over_time.append(self.best_adaptation)

            # Communicate the best solution to other swarms
            if global_best_vector is not None:
                self.receive_best_solution(global_best_vector)

        return self.best_adaptation, self.best_values_over_time

    def generate_population(self, population_size):
        population = []
        for _ in range(population_size):
            new_slime = Slime(self.vector_size, self.a, self.b)
            population.append(new_slime)
        return population

    def calculate_adaptation(self, individual):
        return self.func(individual.vector)

    def update_adaptation(self):
        for slime in self.population:
            adaptation = self.calculate_adaptation(slime)
            if adaptation < slime.best_adaptation:
                slime.best_adaptation = adaptation
                slime.best_vector = copy.deepcopy(slime.vector)
            if adaptation < self.best_adaptation:
                self.best_adaptation = adaptation
                self.best_individual = copy.deepcopy(slime)

    def sample(self, number, exclude_ind):
        population_ind = list(range(len(self.population)))
        population_ind.remove(exclude_ind)
        selected_ind = random.sample(population_ind, number)
        return [self.population[i] for i in selected_ind]

    def sample1(self, number, exclude_ind):
        population_ind = list(range(len(self.population)))
        population_ind.remove(exclude_ind)
        selected_ind = random.sample(population_ind, number)
        return [self.population[i] for i in selected_ind], selected_ind

    def tournament_selection(self, tournament_size=3):
        num_to_replace = int(0.2 * len(self.population))
        selected_particles = []

        # Perform tournament selection to select the best particles
        for _ in range(num_to_replace):
            tournament = random.sample(self.population, tournament_size)
            best_in_tournament = min(tournament, key=lambda p: self.func(p.vector))
            selected_particles.append(copy.deepcopy(best_in_tournament))

        # Replace the worst particles in the population with the selected particles
        self.population.sort(key=lambda p: self.func(p.vector), reverse=True)
        for i in range(num_to_replace):
            new_slime = selected_particles[i]
            new_slime.vector = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
            self.population[-(i + 1)] = new_slime

    def communicate_best_solution(self):
        # Communicate the best solution found so far to other swarms
        global_best_vector = self.best_individual.vector
        for slime in self.population:
            if self.func(global_best_vector) < self.func(slime.vector):
                slime.vector = global_best_vector.copy()

    def receive_best_solution(self, global_best_vector):
        # Receive the best solution from other swarms
        if self.func(global_best_vector) < self.func(self.best_individual.vector):
            self.best_individual.vector = global_best_vector.copy()


class MultiSwarmSlimeMouldAlgorithm:
    def __init__(self, num_swarms, population_size, a, b, func, vector_size, iterations, patience=10,
                 communication_interval=10):
        self.num_swarms = num_swarms
        self.swarms = [
            SlimeMouldSwarm(population_size, a, b, func, vector_size, iterations, patience, communication_interval) for
            _ in range(num_swarms)]
        self.global_best_vector = None
        self.global_best_adaptation = float('inf')
        self.best_swarm_index = None

    def optimize(self):
        best_adaptation = float('inf')
        for i, swarm in enumerate(self.swarms):
            adaptation, _ = swarm.optimize(self.global_best_vector)
            if adaptation < best_adaptation:
                best_adaptation = adaptation
                self.best_swarm_index = i
                self.global_best_vector = swarm.best_individual.vector.copy()
        return best_adaptation, self.swarms[self.best_swarm_index].best_values_over_time
