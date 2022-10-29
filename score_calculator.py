import math

import numpy as np
from sklearn import model_selection

import features
from individual import Individual


def map_wavelet_repeats(value: float):
    return math.floor(value * 30)+1  # 1, 1, 2, 3, 4, ..., 30


def map_sigma(value: float):
    return math.floor(value * 16) * 2 + 1  # 1, 3, 5,...,31


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
            results = model_selection.cross_val_score(estimator=classifier,
                                                      X=x,
                                                      y=y,
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
            image_features = np.array([])
            component = features.image_component(image, individual.genome['COLOR'])
            wavelet_repeats = map_wavelet_repeats(individual.genome['WAVELET_REPEATS'])
            if individual.genome['HISTOGRAM'] > 0.5:
                component = features.histogram_equalization(component)

            if individual.genome['DOG'] > 0.5:
                sigma1 = map_sigma(individual.genome['SIGMA1'])
                sigma2 = map_sigma(individual.genome['SIGMA2'])
                component = features.dog(component, sigma1, sigma2)

            for i in range(wavelet_repeats):
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
            individual.score = self.__get_score(np.array(x), np.array(y))
        else:
            individual.score = 0

        return individual
