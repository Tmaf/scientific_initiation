from matplotlib import pyplot as plt
import numpy as np
from Individual import Individual


class Logger:

    def __init__(self, title):
        self.title = title

    def log_generation(self, generation, best_individual: Individual, worst_individual: Individual, population,
                       best_results):
        self.__print_generation_info(generation, best_individual, worst_individual)
        self.__write_best_genome(best_individual)

    def plot_evolution_score(self, best_score, generation):
        plt.close()
        plt.yscale('linear')
        plt.xscale('linear')
        plt.grid(True)
        plt.yticks(np.arange(min(best_score) - 0.5, 1.05, .01))
        plt.xticks(np.arange(1, len(best_score)+1, 1))
        plt.plot(best_score, 'blue', label='Best Scores')
        # plt.plot(worst_score, 'red', label='Worst Scores')
        plt.legend()
        plt.xlabel("Generations")
        plt.ylabel("Score")
        plt.title("Evolution until generation {}".format(generation))
        plt.savefig(f'plots/{self.title}_evolution_score_generation_{generation}.svg', dpi=500)

    def __write_best_genome(self, best_individual: Individual):
        with open(f'log/{self.title}.txt', "a") as file:
            file.write(f'{best_individual.genome}: {best_individual.score}\n')

    def __print_generation_info(self, generation: int, best_individual: Individual, worst_individual: Individual):
        print(f'''
        {self.title}:
            Generation {generation}:
            WORST:  {worst_individual.score}
            BEST:   {best_individual.score}
            GENOME: {best_individual.genome}
        ''')
