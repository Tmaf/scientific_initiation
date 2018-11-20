# import  cv2
import numpy as np

from sklearn import model_selection
from copy import deepcopy

# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from DataAccess.MongoAccess import DataBase
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

from Individual import Individual
from DataAccess.FileLoader import Files
from DataAccess.MongoAccess import DataBase
import features as ftrs
from sklearn.model_selection import KFold
from multiprocessing.pool import ThreadPool

SEED_K_FOLD = 231234
K_SPLITS = 10
kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=SEED_K_FOLD)
database = DataBase()
SCORING = 'roc_auc'     # 'accuracy'
POPULATION_SIZE = 10
GENERATIONS = 20
MUTATION_TAX = 0.3
NUMBER_OF_IMAGES = 40  # -1 for all
DATABASES = [
    "C:\\Users\\tmaf\\Mega\\dataBase\\FL",
    "C:\\Users\\tmaf\\Mega\\dataBase\\CLL",
    # "C:\\Users\\tmaf\\Mega\\dataBase\\MCL"
]
CLASSIFIERS = {
    # "Nearest Neighbors": KNeighborsClassifier(5),
    # "Linear SVM": SVC(kernel="linear", C=0.025),
    # "Linear SVM": SVC(kernel="linear", C=0.025),
    # "RBF SVM": SVC(gamma=2, C=1),
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    # "Decision Tree": DecisionTreeClassifier(max_depth=5),
    # "Random Forest":  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # "Neural Net": MLPClassifier(alpha=1),
    "AdaBoost": AdaBoostClassifier(),
}
THREAD_NUMBERS = 5

pool = ThreadPool(THREAD_NUMBERS)


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

        if individual.genome['HISTOGRAM']:
            component = ftrs.histeq(component)

        if individual.genome['DOG']:
            component = ftrs.dog(component, individual.genome['SIGMA1'], individual.genome['SIGMA2'])

        for i in range(individual.genome['WAVELET_REPEATS']):

            a, h, v, d = ftrs.wavelet(component, individual.genome['WAVELET'])
            if individual.genome['APPROXIMATION']:
                if individual.genome['ENERGY']:
                    image_features = np.append(image_features, np.array([ftrs.energy(a)]))
                if individual.genome['MEAN']:
                    image_features = np.append(image_features, np.array([np.mean(a)]))
                if individual.genome['MEDIAN']:
                    image_features = np.append(image_features, np.array([np.median(a)]))
                if individual.genome['VARIANCE']:
                    image_features = np.append(image_features, np.array([np.var(a)]))
            if individual.genome['HORIZONTAL']:
                if individual.genome['ENERGY']:
                    image_features = np.append(image_features, np.array([ftrs.energy(h)]))
                if individual.genome['MEAN']:
                    image_features = np.append(image_features, np.array([np.mean(h)]))
                if individual.genome['MEDIAN']:
                    image_features = np.append(image_features, np.array([np.median(h)]))
                if individual.genome['VARIANCE']:
                    image_features = np.append(image_features, np.array([np.var(h)]))
            if individual.genome['VERTICAL']:
                if individual.genome['ENERGY']:
                    image_features = np.append(image_features, np.array([ftrs.energy(v)]))
                if individual.genome['MEAN']:
                    image_features = np.append(image_features, np.array([np.mean(v)]))
                if individual.genome['MEDIAN']:
                    image_features = np.append(image_features, np.array([np.median(v)]))
                if individual.genome['VARIANCE']:
                    image_features = np.append(image_features, np.array([np.var(v)]))
            if individual.genome['DIAGONAL']:
                if individual.genome['ENERGY']:
                    image_features = np.append(image_features, np.array([ftrs.energy(d)]))
                if individual.genome['MEAN']:
                    image_features = np.append(image_features, np.array([np.mean(d)]))
                if individual.genome['MEDIAN']:
                    image_features = np.append(image_features, np.array([np.median(d)]))
                if individual.genome['VARIANCE']:
                    image_features = np.append(image_features, np.array([np.var(d)]))

            component = a

        x.append(image_features)
        y.append(cls)

    # print(individual.genome)
    #
    # for i in range(len(x)):
    #     print(x[i], end='  ' + str(y[i]) + '  ' + str(type(x[i])) +' \n')
    # print('\n\n')
    # print(np.shape(x)[2]==0)

    if np.shape(x)[1] != 0:
        individual.score = score(np.array(x), np.array(y))
    else:
        individual.score = 0


def run():
    # generate first population
    population = [Individual() for _ in range(POPULATION_SIZE)]
    for i in population:
        print(i.genome)
    # for each individual, extract features and evaluate
    for generation in range(GENERATIONS):
        print('\nGeneration {}:'.format(generation))

        for individual in population:
            extract_data(individual)

        # select the bests
        database.save_many('individual', [ind.to_json() for ind in population
                                          if database.find_one('individual', {'genome': ind.genome}) is None])
        best_individual = max(population)
        worst_individual = min(population)

        print('BEST: {}'.format(best_individual.score))
        print('WORST: {}'.format(worst_individual.score))

        # generate new population
        population = [best_individual]
        while len(population) < POPULATION_SIZE:
            population.append(deepcopy(best_individual))
            population[-1].mutate(MUTATION_TAX)
        print('BEST: {} '.format(best_individual.genome))


if __name__ == '__main__':
    run()
