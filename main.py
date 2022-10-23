from typing import List
from Individual import Individual
from ScoreCalculator import ScoreCalculator
import features
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from DataAccess.GraphPlot import plot_evolution_score, print_generation_info
from DataAccess.ImageLoader import ImageLoader
from sklearn.model_selection import StratifiedKFold

# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier


def create_population(size: int) -> List[Individual]:
    return [Individual() for _ in range(size)]


def mutate_population(
        population: List[Individual],
        best_individual: Individual,
        worst_individual: Individual,
        mutation_tax: float
):
    for individual in population:
        if individual != best_individual:
            individual.mutate(mutation_tax, best=best_individual, worst=worst_individual)


def make_extract_data(databases, images_number, score_calculator: ScoreCalculator):
    def extract_data(individual: Individual):
        image_loader = ImageLoader(databases)
        x = []
        y = []

        for name, image, cls in image_loader.get_image(class_limit=images_number):
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
            individual.score = score_calculator.get_score(np.array(x), np.array(y))
        else:
            individual.score = 0

        return individual

    return extract_data


class Main:
    def __init__(self,
                 process_numbers: int,
                 population_size: int,
                 generations_number: int,
                 images_number: int,
                 mutation_tax: float,
                 plot_name: str,
                 databases: List[str],
                 score_calculator: ScoreCalculator):
        self.databases = databases
        self.process_numbers = process_numbers
        self.population_size = population_size
        self.images_number = images_number
        self.mutation_tax = mutation_tax
        self.generations_number = generations_number
        self.plot_name = plot_name
        self.score_calculator = score_calculator

    def run(self):
        pool = Pool(self.process_numbers)
        extract_data = make_extract_data(self.databases, self.images_number, self.score_calculator)
        population = create_population(self.population_size)
        best_results: List = []
        worst_results: List = []

        for generation in range(self.generations_number):
            population = pool.map(extract_data, population)
            best_individual = max(population)
            worst_individual = min(population)
            print_generation_info(generation, best_individual, worst_individual)
            best_results.append(best_individual.score)
            worst_results.append(worst_individual.score)
            mutate_population(population, best_individual, worst_individual, self.mutation_tax)
        plot_evolution_score(best_results, worst_results, self.generations_number, self.plot_name)


if __name__ == '__main__':
    PROCESS_NUMBERS = 12
    POPULATION_SIZE = 30
    NUMBER_OF_GENERATIONS = 50
    MUTATION_TAX = 0.2
    NUMBER_OF_IMAGES = 50
    PLOT_NAME = "individuals"
    SEED_K_FOLD = 123456
    K_SPLITS = 10
    SCORING = 'roc_auc'  # 'accuracy'
    DATABASES = [
        "data/FL",
        # "data/CLL",
        "data/MCL"
    ]
    CLASSIFIERS = {
        # "Nearest Neighbors": lambda:  KNeighborsClassifier(5),
        # "Linear SVM": lambda: SVC(kernel="linear", C=0.025),
        # "Sigmoid SVM": lambda: SVC(kernel="sigmoid", C=0.025),
        # "RBF SVM": lambda: SVC(kernel="rbf", C=0.025),
        # "Gaussian Process": lambda:  GaussianProcessClassifier(1.0 * RBF(1.0)),
        # "Decision Tree": lambda:  DecisionTreeClassifier(max_depth=6),
        # "Random Forest": lambda: RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # "Neural Net": lambda: MLPClassifier(alpha=1),
        "AdaBoost": lambda: AdaBoostClassifier(),
    }

    stratified_k_fold = StratifiedKFold(n_splits=K_SPLITS,
                                        shuffle=True,
                                        random_state=SEED_K_FOLD
                                        )

    scoreCalculator = ScoreCalculator(cross_validation_strategy=stratified_k_fold,
                                      scoring=SCORING,
                                      classifiers=CLASSIFIERS)

    main = Main(process_numbers=PROCESS_NUMBERS,
                population_size=POPULATION_SIZE,
                generations_number=NUMBER_OF_GENERATIONS,
                images_number=NUMBER_OF_IMAGES,
                mutation_tax=MUTATION_TAX,
                plot_name=PLOT_NAME,
                databases=DATABASES,
                score_calculator=scoreCalculator)
    main.run()
