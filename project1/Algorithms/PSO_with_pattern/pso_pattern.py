import copy
import random
from project1.Algorithms.PSO import particle


class Pso:
    def __init__(self, particles_num, a, b, func, inertia, cognitive_constant, social_constant, vector_size,
                 iterations, cr=0.75, f=0.5, pm=0.6, patience=10):
        self.a = a
        self.b = b
        self.func = func
        self.inertia = inertia
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.vector_size = vector_size
        self.iterations = iterations
        self.population = self.generate_population(particles_num)
        self.best_adaptation = float('inf')
        self.best_particle = None
        self.best_values_over_time = []
        self.cr = cr
        self.f = f
        self.pm = pm
        self.patience = patience
        self.stagnation_counter = 0

    def optimize(self):
        for i in range(self.iterations):
            prev_best_adaptation = self.best_adaptation
            self.update_adaptation()
            if self.best_adaptation == prev_best_adaptation:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter >= self.patience:
                self.tournament_selection()
                self.stagnation_counter = 0  # Reset the counter after diversification

            for j in range(len(self.population)):
                self.population[j].update_velocity(self.best_particle)
                self.population[j].update_position()
                o = particle.Particle(self.vector_size, self.inertia, self.cognitive_constant,
                                      self.social_constant, self.a, self.b)
                for k in range(len(self.population[j].vector)):
                    d, ind_d = self.sample(1, j)  # randomly selected particle
                    if self.func(self.population[j].best_vector) < self.func(
                            self.population[ind_d[0]].best_vector):  # crossover
                        r = random.uniform(0, 1)
                        o.vector[k] = r * self.population[j].best_vector[k] + (
                                (1 - r) * self.best_particle.best_vector[k])
                    else:
                        o.vector[k] = self.population[ind_d[0]].vector[k]

                for k in range(len(self.population[j].vector)):  # mutation
                    if random.uniform(0, 1) < self.pm:
                        o.vector[k] = random.uniform(self.a, self.b)

                o_adaptation = self.func(o.vector)
                if o_adaptation < self.func(self.population[j].vector):
                    self.population[j] = o
                if o_adaptation < self.best_adaptation:
                    self.best_adaptation = o_adaptation
                    self.best_particle = o  # Update the best particle reference
            self.best_values_over_time.append(self.best_adaptation)
        return self.best_adaptation, self.best_values_over_time

    def generate_population(self, particles_num):
        particles = []
        for i in range(particles_num):
            new_particle = particle.Particle(self.vector_size, self.inertia, self.cognitive_constant,
                                             self.social_constant, self.a, self.b)
            particles.append(new_particle)
        return particles

    def calculate_adaptation(self, particlee):
        return self.func(particlee.vector)

    def update_adaptation(self):
        for particlee in self.population:
            adaptation = self.calculate_adaptation(particlee)
            if adaptation < particlee.best_adaptation:  # local
                particlee.best_adaptation = adaptation
                particlee.best_vector = copy.deepcopy(particlee.vector)
            if adaptation < self.best_adaptation:  # global
                self.best_adaptation = adaptation
                self.best_particle = particlee

    def sample(self, number, exclude_ind):
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
            new_particle = selected_particles[i]
            new_particle.vector = [random.uniform(self.a, self.b) for _ in range(self.vector_size)]
            self.population[-(i + 1)] = new_particle
