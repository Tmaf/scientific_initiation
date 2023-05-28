from matplotlib import pyplot as plt
import numpy as np
from individual import Individual
import pandas as pd


class Logger:

    def __init__(self, title):
        self.title = title
        with open(f'results/log/{self.title}.txt', "w") as file:
            file.close()


    def log_generation(self, generation, best_individual: Individual, worst_individual: Individual, population,
                       best_results):
        self.__print_generation_info(generation, best_individual, worst_individual)
        self.__write_best_genome(best_individual, generation)

    def plot_evolution_score(self, best_score, generation):
        plt.close()
        plt.yscale('linear')
        plt.xscale('linear')
        plt.grid(True)
        plt.yticks(np.arange(min(best_score) - 0.5, 1.05, .01))
        plt.xticks(np.arange(0, len(best_score), 1))
        plt.plot(best_score, 'blue', label='Best Scores')
        # plt.plot(worst_score, 'red', label='Worst Scores')
        plt.legend()
        plt.xlabel("Generations")
        plt.ylabel("Score")
        plt.title("Evolution until generation {}".format(generation))
        plt.savefig(f'results/plots/{self.title}_evolution_score_generation_{generation}.svg', dpi=500)

    def __write_best_genome(self, best_individual: Individual, generation):
        with open(f'results/log/{self.title}.txt', "a") as file:
            file.write(f'{best_individual.genome}: {best_individual.score}\n')
        dataframe = pd.DataFrame(best_individual.image_table)
        dataframe.to_csv(f'results/features/{self.title}_generation_{generation}.csv', index=False, header=False)

    def __print_generation_info(self, generation: int, best_individual: Individual, worst_individual: Individual):
        print(f'''
        {self.title}:
            Generation {generation}:
            WORST:  {worst_individual.score}
            GENOME:  {worst_individual.genome}
            BEST:   {best_individual.score}
            GENOME: {best_individual.genome}
        ''')
