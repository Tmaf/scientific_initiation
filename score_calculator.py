from typing import List

import numpy as np
from sklearn import model_selection

import features
from individual import Individual


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
            results = model_selection.cross_val_score(classifier, x, y,
                                                      cv=self.cross_validation_strategy,
                                                      scoring=self.scoring)
            scores = np.append(scores, [results.mean()])
        return scores.mean()

    def execute(self, individual: Individual):
        x = []
        y = []

        for name, image, cls in self.image_loader.get_next():
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
            individual.score = self.__get_score(np.array(x), np.array(y))
        else:
            individual.score = 0

        return individual
