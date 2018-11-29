import random as rd

FEATURES = dict()

FEATURES['COLOR'] = ['r', 'g', 'b', 'h', 's', 'v', 'y']
FEATURES['SIGMA1'] = [1, 3, 5, 7, 9, 11, 13]
FEATURES['SIGMA2'] = [1, 3, 5, 7, 9, 11, 13]
FEATURES['WAVELET'] = ['db2', 'db4', 'db8', 'db16', 'db32']
FEATURES['WAVELET_REPEATS'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

FEATURES['HISTOGRAM'] = [True, False]
FEATURES['DOG'] = [True, False]
FEATURES['APPROXIMATION'] = [True, False]
FEATURES['HORIZONTAL'] = [True, False]
FEATURES['VERTICAL'] = [True, False]
FEATURES['DIAGONAL'] = [True, False]

FEATURES['ENERGY'] = [True, False]
FEATURES['MEAN'] = [True, False]
FEATURES['MEDIAN'] = [True, False]
FEATURES['VARIANCE'] = [True, False]
FEATURES['MODE'] = [True, False]

FEATURES['ENTROPY'] = [True, False]
FEATURES['KURTOSIS'] = [True, False]


class Individual(object):

    def __init__(self, genome=None):
        if not genome:
            self.__genome = dict()
            self.__genome['COLOR'] = rd.choice(FEATURES['COLOR'])
            self.__genome['SIGMA1'] = rd.choice(FEATURES['SIGMA1'])
            self.__genome['SIGMA2'] = rd.choice(FEATURES['SIGMA2'])
            self.__genome['WAVELET'] = rd.choice(FEATURES['WAVELET'])
            self.__genome['WAVELET_REPEATS'] = rd.choice(FEATURES['WAVELET_REPEATS'])

            self.__genome['HISTOGRAM'] = rd.random()
            self.__genome['DOG'] = rd.random()
            self.__genome['HORIZONTAL'] = rd.random()
            self.__genome['VERTICAL'] = rd.random()
            self.__genome['DIAGONAL'] = rd.random()
            self.__genome['APPROXIMATION'] = rd.random()
            self.__genome['ENERGY'] = rd.random()
            self.__genome['MEAN'] = rd.random()
            self.__genome['MEDIAN'] = rd.random()
            self.__genome['VARIANCE'] = rd.random()
            # self.__genome['MODE'] = rd.random()

            # self.__genome['ENTROPY'] = rd.choice(FEATURES['ENTROPY'])
            # self.__genome['KURTOSIS'] = rd.choice(FEATURES['KURTOSIS'])
        else:
            self.__genome = genome
        self.__score = 0

    @property
    def genome(self):
        return self.__genome

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, score):
        self.__score = score

    @genome.setter
    def genome(self, genome):
        self.__genome = genome

    def mutate(self, tax, best, worst):
        self.__score = 0
        for feature in self.__genome:
            if feature not in ['COLOR', 'SIGMA1', 'SIGMA2', 'WAVELET_REPEATS', 'WAVELET']:
                self.genome[feature] = (self.genome[feature]
                                        + (rd.random() * (best.genome[feature] - self.genome[feature]))
                                        - (rd.random() * (worst.genome[feature] - self.genome[feature])))
                if rd.random() < tax:
                    self.genome[feature] += rd.random()
                if rd.random() < tax:
                    self.genome[feature] -= rd.random()

            else:
                if rd.random() < tax:
                    if rd.random() < tax:
                        self.genome[feature] = rd.choice(FEATURES[feature])
                    else:
                        self.genome[feature] = best.genome[feature]

    def random_mutate(self, tax):
        self.__score = 0
        for feature in self.__genome:
            if rd.random() < tax:
                if feature not in ['COLOR', 'SIGMA1', 'SIGMA2', 'WAVELET_REPEATS', 'WAVELET']:
                    self.genome[feature] = rd.choice(FEATURES[feature])
                else:
                    self.genome[feature] = rd.random()

    def to_json(self, additional_info=()):
        genome = self.__genome
        score = self.__score
        json = {
            'genome': genome,
            'score': score
        }
        for i in additional_info:
            json[i] = additional_info[i]
        return json

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score
