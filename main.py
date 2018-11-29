# import  cv2
import numpy as np

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from DataAccess.MongoAccess import DataBase
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
from Individual import Individual
from DataAccess.GraphPlot import plot_generation_score
from DataAccess.GraphPlot import plot_evolution_score
from DataAccess.FileLoader import Files
from DataAccess.MongoAccess import DataBase
import features as ftrs
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool

SEED_K_FOLD = 231234
K_SPLITS = 4
kf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=SEED_K_FOLD)
database = DataBase()
SCORING = 'roc_auc'     # 'accuracy'
POPULATION_SIZE = 10
GENERATIONS = 20
MUTATION_TAX = 0.2
NUMBER_OF_IMAGES = 20  # -1 for all
DATABASES = [
    "C:\\Users\\tmaf\\Mega\\dataBase\\FL",
    # "C:\\Users\\tmaf\\Mega\\dataBase\\CLL",
    "C:\\Users\\tmaf\\Mega\\dataBase\\MCL"
]
CLASSIFIERS = {
    # "Nearest Neighbors": KNeighborsClassifier(5),
    # "Linear SVM": SVC(kernel="linear", C=0.025),
    # "Sigmoid SVM": SVC(kernel="sigmoid", C=0.025),
    # "RBF SVM": SVC(kernel="rbf", C=0.025),
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    # "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest":  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # "Neural Net": MLPClassifier(alpha=1),
    # "AdaBoost": AdaBoostClassifier(),
}
PROCESS_NUMBERS = 4
DATABASE_NAME = "individuals"


def score(x, y):
    scores = np.array([])
    for model in CLASSIFIERS:
        results = model_selection.cross_val_score(CLASSIFIERS[model], x, y, cv=kf, scoring=SCORING)
        # print('-----' + model)
        # print("Accuracy: " + str(results.mean()) + " (" + str(results.std()) + ")")
        scores = np.append(scores, [results.mean()])
    return scores.mean()


def extract_data(individual):
    images = Files(DATABASES)
    x = []
    y = []

    for name, image, cls in images.get_image(class_limit=NUMBER_OF_IMAGES):
        image_features = np.array([])
        component = ftrs.channel(image, individual.genome['COLOR'])

        if individual.genome['HISTOGRAM'] > 0.5:
            component = ftrs.histeq(component)

        if individual.genome['DOG'] > 0.5:
            component = ftrs.dog(component, individual.genome['SIGMA1'], individual.genome['SIGMA2'])

        for i in range(individual.genome['WAVELET_REPEATS']):
            a, h, v, d = ftrs.wavelet(component, individual.genome['WAVELET'])
            if individual.genome['APPROXIMATION'] > 0.5:
                if individual.genome['ENERGY'] > 0.5:
                    image_features = np.append(image_features, np.array([ftrs.energy(a)]))
                if individual.genome['MEAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.mean(a)]))
                if individual.genome['MEDIAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.median(a)]))
                if individual.genome['VARIANCE'] > 0.5:
                    image_features = np.append(image_features, np.array([np.var(a)]))
                # if individual.genome['MODE'] > 0.5:
                #     image_features = np.append(image_features, np.array([np.argmax(np.bincount(a))]))
            if individual.genome['HORIZONTAL'] > 0.5:
                if individual.genome['ENERGY'] > 0.5:
                    image_features = np.append(image_features, np.array([ftrs.energy(h)]))
                if individual.genome['MEAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.mean(h)]))
                if individual.genome['MEDIAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.median(h)]))
                if individual.genome['VARIANCE'] > 0.5:
                    image_features = np.append(image_features, np.array([np.var(h)]))
                # if individual.genome['MODE'] > 0.5:
                #     image_features = np.append(image_features, np.array([np.argmax(np.bincount(h))]))
            if individual.genome['VERTICAL'] > 0.5:
                if individual.genome['ENERGY'] > 0.5:
                    image_features = np.append(image_features, np.array([ftrs.energy(v)]))
                if individual.genome['MEAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.mean(v)]))
                if individual.genome['MEDIAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.median(v)]))
                if individual.genome['VARIANCE'] > 0.5:
                    image_features = np.append(image_features, np.array([np.var(v)]))
                # if individual.genome['MODE'] > 0.5:
                #     image_features = np.append(image_features, np.array([np.argmax(np.bincount(v))]))
            if individual.genome['DIAGONAL'] > 0.5:
                if individual.genome['ENERGY'] > 0.5:
                    image_features = np.append(image_features, np.array([ftrs.energy(d)]))
                if individual.genome['MEAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.mean(d)]))
                if individual.genome['MEDIAN'] > 0.5:
                    image_features = np.append(image_features, np.array([np.median(d)]))
                if individual.genome['VARIANCE'] > 0.5:
                    image_features = np.append(image_features, np.array([np.var(d)]))
                # if individual.genome['MODE'] > 0.5:
                #     image_features = np.append(image_features, np.array([np.argmax(np.bincount(d))]))

            component = a

        x.append(image_features)
        y.append(cls)

    if np.shape(x)[1] != 0:
        individual.score = score(np.array(x), np.array(y))
    else:
        individual.score = 0

    return individual


def run():
    pool = Pool(PROCESS_NUMBERS)
    best_results = []
    worst_results = []
    # generate first population
    population = [Individual() for _ in range(POPULATION_SIZE)]
    # for each individual, extract features and evaluate
    for generation in range(GENERATIONS):
        print('\nGeneration {}:'.format(generation))

        for i in population:
            print(i.genome)

        population = pool.map(extract_data, population)

        # select the bests
        database.save_many(DATABASE_NAME, [ind.to_json({'generation': generation}) for ind in population
                                           if database.find_one(DATABASE_NAME, {'genome': ind.genome}) is None])
        best_individual = max(population)
        worst_individual = min(population)

        print('BEST: {}'.format(best_individual.score))
        print('WORST: {}'.format(worst_individual.score))
        print('BEST:{} \n {} '.format(best_individual, best_individual.genome))

        best_results.append(best_individual.score)
        worst_results.append(worst_individual.score)

        if generation % 10 == 0 and generation > 0:
            plot_evolution_score(best_results, worst_results, generation, DATABASE_NAME)
        if generation % 5 == 0 and generation > 0:
            plot_generation_score([i.score for i in population], generation, DATABASE_NAME)

        for individual in population:
            if individual != best_individual:
                individual.mutate(MUTATION_TAX, best=best_individual, worst=worst_individual)


if __name__ == '__main__':
    run()
