import particle


class Pso:
    def __init__(self, particles_num, a, b, func, inertia, cognitive_constant, social_constant, vector_size,
                 iterations):
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
        self.best_particle = self.population[0]

    def generate_population(self, particles_num):
        particles = []
        for i in range(particles_num):
            new_particle = particle.Particle(self.vector_size, self.inertia, self.cognitive_constant,
                                             self.social_constant, self.a, self.b)
            particles.append(new_particle)
        return particles

    def calculate_adaptation(self, particle):
        return self.func(particle.vector)

    def update_adaptation(self):
        for particle in self.population:
            adaptation = self.calculate_adaptation(particle)
            if adaptation < particle.best_adaptation:
                particle.best_adaptation = adaptation
                particle.best_vector = particle.vector
            if adaptation < self.best_adaptation:
                self.best_adaptation = adaptation
                self.best_particle = particle





