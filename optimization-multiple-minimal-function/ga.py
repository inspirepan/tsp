import numpy as np
from arguments import N


class GeneticAlgorithms:
    def solve(self):
        p = self.get_initial_generation(N)  # 种群P
        k = 0  # 当前代
        s = 0
        f_v = self.get_fitness_value(p)
        while s > 0:
            m = 0
            while m < N:
                temp = np.random.choice(p, 2)
                if self.p_crossover(temp) > np.random.uniform(0, 1):
                    temp = self.crossover(temp)
                mutated = self.mutation(temp)
                p.extend(mutated)
                m += 2
            k += 1
            f_v = self.get_fitness_value(p)

    def get_fitness_value(self):
        pass

    def get_initial_generation(self, temp):
        pass

    def p_crossover(self, temp):
        pass

    def crossover(self, temp):
        pass

    def mutation(self, temp):
        pass
