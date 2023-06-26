import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

RANDOM_STATE = 123456


def plot_values(path):
    data = []
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            value = line.split("}: ")[-1]
            data.append(float(value))
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.yscale('linear')
    plt.xscale('linear')
    plt.grid(True)
    # plt.yticks(np.arange(min(data), max(), 0.01))
    plt.xticks(np.arange(0, len(data) + 2, 2))
    ax.set(
        xlabel="Geração",
        ylabel="Score\n",
        title=f"Score",
    )
    plt.tight_layout()
    plt.plot(data, color='blue', label='Melhores Indivíduos de cada geração')
    plt.legend(loc='lower right')

    plt.show()


def plotMultiplesValues(values,title ):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.yscale('linear')
    plt.xscale('linear')
    # plt.grid(True)
    # plt.yticks(np.arange(min(data), max(), 0.01))
    plt.xticks(np.arange(0, 22, 2))
    ax.set(
        xlabel="Geração",
        ylabel="Score",
        title=f"Evolução {title}",
    )
    for (tlabel,color,path) in values:
        with open(path) as file:
            lines = file.readlines()
            data = []
            for line in lines:
                value = line.split("}: ")[-1]
                data.append(float(value))
            plt.plot(data,color=color, label=tlabel, linestyle="-", marker=".", linewidth=.5)
            # ax.scatter(len(data)-1, data[-1], c=color, marker=".", label='{:.2f}%'.format(data[-1]*100))
            # ax.scatter(0, data[0], c=color, marker=".",  label='{:.2f}%'.format(data[0]*100))


    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":

    experiments = [
        ("LLC x LF", "FL_CLL"),
        ("LLC x LCM", "CLL_MCL"),
        ("LF x LCM", "FL_MCL" )
    ]

    for (title, fold) in experiments:
        plotMultiplesValues([
            ("Random Forest","tab:blue", f"../results/{fold}_RANDOM_FOREST/best.txt"),
            ("Ada Boost","tab:orange", f"../results/{fold}_ADA_BOOST/best.txt"),
            ("SVM", "tab:green", f"../results/{fold}_LINEAR_SVM/best.txt")
        ],title=title)
