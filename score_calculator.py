import math

import numpy as np
from sklearn import model_selection, preprocessing

import features
from individual import Individual


def map_wavelet_repeats(value: float):
    return math.floor(value * 30)+1  # 1, 1, 2, 3, 4, ..., 30


def map_sigma(value: float):
    return math.floor(value * 16) * 2 + 1  # 1, 3, 5,...,31


def apply_descriptors(individual, image_features):
    descriptors = np.empty([])

    if individual.genome['ENERGY'] > 0.5:
        descriptors = np.append(descriptors, np.array([features.energy(image_features)]))
    if individual.genome['MEAN'] > 0.5:
        descriptors = np.append(descriptors, np.array([np.mean(image_features)]))
    if individual.genome['MEDIAN'] > 0.5:
        descriptors = np.append(descriptors, np.array([np.median(image_features)]))
    if individual.genome['VARIANCE'] > 0.5:
        descriptors = np.append(descriptors, np.array([np.var(image_features)]))
    return descriptors


class ScoreCalculator:
    def __init__(self, cross_validation_strategy, scoring, classifiers, image_loader):
        self.cross_validation_strategy = cross_validation_strategy
        self.scoring = scoring
        self.classifiers = classifiers
        self.image_loader = image_loader

    def __get_score(self, x, y):
        scores = np.array([])
        for model in self.classifiers:
            classifier = self.classifiers[model]()
            X = preprocessing.scale(x)
            Y = preprocessing.scale(y)
            results = model_selection.cross_val_score(estimator=classifier,
                                                      X=X,
                                                      y=Y,
                                                      cv=self.cross_validation_strategy,
                                                      scoring=self.scoring,
                                                      n_jobs=10
                                                      )
            scores = np.append(scores, [results.mean()])
        return scores.mean()

    def execute(self, individual: Individual):
        x = []
        y = []

        for name, image, cls in self.image_loader.get_next():
            component = features.image_component(image, individual.genome['COLOR'])
            wavelet_repeats = map_wavelet_repeats(individual.genome['WAVELET_REPEATS'])
            if individual.genome['HISTOGRAM'] > 0.5:
                component = features.histogram_equalization(component)

            if individual.genome['DOG'] > 0.5:
                sigma1 = map_sigma(individual.genome['SIGMA1'])
                sigma2 = map_sigma(individual.genome['SIGMA2'])
                component = features.dog(component, sigma1, sigma2)

            image_features = np.array([])
            for i in range(wavelet_repeats):
                a, h, v, d = features.wavelet(component, individual.genome['WAVELET'])

                if individual.genome['APPROXIMATION'] > 0.5:
                    image_features =  np.append(image_features,apply_descriptors(individual,a))

                if individual.genome['HORIZONTAL'] > 0.5:
                    image_features = np.append(image_features, apply_descriptors(individual, h))

                if individual.genome['VERTICAL'] > 0.5:
                    image_features = np.append(image_features, apply_descriptors(individual, v))

                if individual.genome['DIAGONAL'] > 0.5:
                    image_features = np.append(image_features, apply_descriptors(individual, d))

                component = a

            x.append(image_features)
            y.append(cls)

        if np.shape(x)[1] != 0:
            individual.score = self.__get_score(np.array(x), np.array(y))
            individual.image_table = np.append(np.array(x),(np.array(y).reshape(-1, 1)),axis=1)
        else:
            individual.score = 0

        return individual
