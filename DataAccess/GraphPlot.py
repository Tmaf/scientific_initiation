from matplotlib import pyplot as plt
import numpy as np
from Individual import Individual


def plot_evolution_score(best_score, worst_score, generation, data_base):
    plt.close()
    plt.yscale('linear')
    plt.xscale('linear')
    plt.yticks(np.arange(min(best_score)-0.5, 1.05, .01))
    plt.xticks(np.arange(0, len(best_score), 1))
    plt.grid(True)
    plt.plot(best_score, 'blue', label='Best Scores')
    # plt.plot(worst_score, 'red', label='Worst Scores')
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Score")
    plt.title("Evolution until generation {}".format(generation))
    plt.savefig('plots/{}_evolution_{}.svg'.format(data_base, generation), dpi=500)


def plot_generation_score(array, generation, data_base):
    plt.close()
    plt.bar([i for i in range(len(array))], array, width=0.3)
    plt.xlabel("Individuals")
    plt.ylabel("Score")
    plt.title("Scores in generation {}".format(generation))
    plt.savefig('plots/{}_generation_{}.png'.format(data_base, generation))


def print_generation_info(generation: int, best_individual: Individual, worst_individual: Individual):
    print(f'''
    Generation {generation}:
        WORST:  {worst_individual.score}
        BEST:   {best_individual.score}
        GENOME: {best_individual.genome}
    ''')
