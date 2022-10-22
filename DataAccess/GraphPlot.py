from matplotlib import pyplot as plt


def plot_evolution_score(best_score, worst_score, generation, data_base):
    plt.close()
    plt.plot(best_score, 'blue', label='Best Scores')
    plt.plot(worst_score, 'red', label='Worst Scores')
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("Score")
    plt.title("Evolution until generation {}".format(generation))
    plt.savefig('plots/{}_evolution_{}.png'.format(data_base, generation))


def plot_generation_score(array, generation, data_base):
    plt.close()
    plt.bar([i for i in range(len(array))], array, width=0.3)
    plt.xlabel("Individuals")
    plt.ylabel("Score")
    plt.title("Scores in generation {}".format(generation))
    plt.savefig('plots/{}_generation_{}.png'.format(data_base, generation))
