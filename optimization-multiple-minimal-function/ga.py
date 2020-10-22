import numpy as np
from arguments import N


class GeneticAlgorithms:
    def solve(self):
        # 种群P
        p = self.get_initial_generation(N)
        # 当前代
        k = 0
        s = 0
        while s > 0:
            m = 0
            while m < N:
                np.random.shuffle(p)
                temp = p[:2]
                if self.p_crossover(temp) > np.random.uniform(0, 1):
                    temp = self.crossover(temp)
                mutated = self.mutation(temp)
                m += 2
            k += 1

    def get_fitness_value(self):
        pass

    def get_initial_generation(self):
        pass

    def p_crossover(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass
