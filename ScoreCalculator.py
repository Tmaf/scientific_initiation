import numpy as np
from sklearn import model_selection


class ScoreCalculator:
    def __init__(self, cross_validation_strategy, scoring, classifiers):
        self.cross_validation_strategy = cross_validation_strategy
        self.scoring = scoring
        self.classifiers = classifiers

    def get_score(self, x, y):
        scores = np.array([])
        for model in self.classifiers:
            classifier = self.classifiers[model]()
            results = model_selection.cross_val_score(classifier, x, y,
                                                      cv=self.cross_validation_strategy,
                                                      scoring=self.scoring)
            scores = np.append(scores, [results.mean()])
        return scores.mean()
