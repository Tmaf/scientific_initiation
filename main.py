# import  cv2
import numpy as np
from sklearn import model_selection
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
# from DataAccess.GraphPlot import plot_generation_score
from DataAccess.GraphPlot import plot_evolution_score
from DataAccess.FileLoader import Files
# from DataAccess.MongoAccess import DataBase
import features
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool

SEED_K_FOLD = 231234
K_SPLITS = 4
kf = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=SEED_K_FOLD)
# database = DataBase()
SCORING = 'roc_auc'  # 'accuracy'
POPULATION_SIZE = 10
NUMBER_OF_GENERATIONS = 30
MUTATION_TAX = 0.2
NUMBER_OF_IMAGES = 30  # -1 for all
DATABASES = [
    "data/FL",
    # "data/CLL",
    "data/MCL"
]
CLASSIFIERS = {
    # "Nearest Neighbors": KNeighborsClassifier(5),
    # "Linear SVM": SVC(kernel="linear", C=0.025),
    # "Sigmoid SVM": SVC(kernel="sigmoid", C=0.025),
    # "RBF SVM": SVC(kernel="rbf", C=0.025),
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    # "Decision Tree": DecisionTreeClassifier(max_depth=6),
    # "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # "Neural Net": MLPClassifier(alpha=1),
    "AdaBoost": AdaBoostClassifier(),
}
PROCESS_NUMBERS = 8
DATABASE_NAME = "individuals"


def score(x, y):
    scores = np.array([])
    for model in CLASSIFIERS:
        results = model_selection.cross_val_score(CLASSIFIERS[model], x, y, cv=kf, scoring=SCORING)
        scores = np.append(scores, [results.mean()])
    return scores.mean()


def extract_data(individual):
    images = Files(DATABASES)
    x = []
    y = []

    for name, image, cls in images.get_image(class_limit=NUMBER_OF_IMAGES):
        image_features = np.array([])
        component = features.channel(image, individual.genome['COLOR'])

        if individual.genome['HISTOGRAM'] > 0.5:
            component = features.histeq(component)

        if individual.genome['DOG'] > 0.5:
            component = features.dog(component, individual.genome['SIGMA1'], individual.genome['SIGMA2'])

        for i in range(individual.genome['WAVELET_REPEATS']):
            a, h, v, d = features.wavelet(component, individual.genome['WAVELET'])
            if individual.genome['APPROXIMATION'] > 0.5:
                if individual.genome['ENERGY'] > 0.5:
                    image_features = np.append(image_features, np.array([features.energy(a)]))
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
                    image_features = np.append(image_features, np.array([features.energy(h)]))
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
                    image_features = np.append(image_features, np.array([features.energy(v)]))
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
                    image_features = np.append(image_features, np.array([features.energy(d)]))
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


def print_generation_info(generation, best_individual, worst_individual):
    print(f'''
    Generation {generation}:
        WORST:  {worst_individual.score}
        BEST:   {best_individual.score}
        GENOME: {best_individual.genome}
    ''')


def create_population():
    return [Individual() for _ in range(POPULATION_SIZE)]


def mutate_population(best_individual, population, worst_individual):
    for individual in population:
        if individual != best_individual:
            individual.mutate(MUTATION_TAX, best=best_individual, worst=worst_individual)


def run():
    pool = Pool(PROCESS_NUMBERS)
    best_results = []
    worst_results = []
    population = create_population()
    for generation in range(0, NUMBER_OF_GENERATIONS + 1):
        population = pool.map(extract_data, population)
        best_individual = max(population)
        worst_individual = min(population)
        print_generation_info(generation, best_individual, worst_individual)
        best_results.append(best_individual.score)
        worst_results.append(worst_individual.score)
        mutate_population(best_individual, population, worst_individual)
    plot_evolution_score(best_results, worst_results, NUMBER_OF_GENERATIONS, DATABASE_NAME)


if __name__ == '__main__':
    run()
