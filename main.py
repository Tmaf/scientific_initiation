from Jaya import Jaya
from ScoreCalculator import ScoreCalculator
from data_access import Logger, ImageLoader
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    PROCESS_NUMBERS = 12
    POPULATION_SIZE = 10
    NUMBER_OF_GENERATIONS = 20
    MUTATION_TAX = 0.2
    NUMBER_OF_IMAGES = 20
    PLOT_NAME = "FL_MCL_RANDOM_FOREST"
    SEED_K_FOLD = 123456
    K_SPLITS = 4
    SCORING = 'roc_auc'  # 'accuracy'
    DATABASES = [
        "data/FL",
        "data/CLL",
        # "data/MCL"
    ]
    CLASSIFIERS = {
        # "Nearest Neighbors": lambda:  KNeighborsClassifier(5),
        # "Linear SVM": lambda: SVC(kernel="linear", C=0.025),
        # "Sigmoid SVM": lambda: SVC(kernel="sigmoid", C=0.025),
        # "RBF SVM": lambda: SVC(kernel="rbf", C=0.025),
        # "Gaussian Process": lambda:  GaussianProcessClassifier(1.0 * RBF(1.0)),
        # "Decision Tree": lambda:  DecisionTreeClassifier(max_depth=6),
        "Random Forest": lambda: RandomForestClassifier(max_depth=5,
                                                        n_estimators=10,
                                                        max_features=1,
                                                        random_state=SEED_K_FOLD
                                                        ),
        # "Neural Net": lambda: MLPClassifier(alpha=1),
        # "AdaBoost": lambda: AdaBoostClassifier(),
    }

    logger = Logger(title=PLOT_NAME)
    poolMapper = Pool(PROCESS_NUMBERS)
    image_loader = ImageLoader(directories_path=DATABASES,
                               images_number=NUMBER_OF_IMAGES
                               )
    stratified_k_fold = StratifiedKFold(n_splits=K_SPLITS,
                                        shuffle=True,
                                        random_state=SEED_K_FOLD
                                        )
    scoreCalculator = ScoreCalculator(image_loader=image_loader,
                                      cross_validation_strategy=stratified_k_fold,
                                      scoring=SCORING,
                                      classifiers=CLASSIFIERS
                                      )
    jaya = Jaya(population_size=POPULATION_SIZE,
                generations_number=NUMBER_OF_GENERATIONS,
                mutation_tax=MUTATION_TAX,
                score_calculator=scoreCalculator,
                logger=logger,
                pool=poolMapper
                )

    jaya.execute()

# TODO:
# 5. Create a Classifiers abstraction
# 3. Pensar em uma implementação otimizada do ImageLoader
# 4. Ajustar Individual pra permitir o jaya em mais genes
