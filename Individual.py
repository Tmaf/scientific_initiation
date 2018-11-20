import random as rd

FEATURES = dict()

FEATURES['COLOR'] = ['r', 'g', 'b', 'h', 's', 'v', 'y']
FEATURES['HISTOGRAM'] = [True, False]
FEATURES['DOG'] = [True, False]
FEATURES['SIGMA1'] = [1, 3, 5, 7, 9, 11, 13]
FEATURES['SIGMA2'] = [1, 3, 5, 7, 9, 11, 13]
FEATURES['WAVELET'] = ['db2', 'db4', 'db8']
FEATURES['WAVELET_REPEATS'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, ]
FEATURES['APPROXIMATION'] = [True, False]
FEATURES['HORIZONTAL'] = [True, False]
FEATURES['VERTICAL'] = [True, False]
FEATURES['DIAGONAL'] = [True, False]

FEATURES['ENERGY'] = [True, False]
FEATURES['ENTROPY'] = [True, False]
FEATURES['MEAN'] = [True, False]
FEATURES['MEDIAN'] = [True, False]
FEATURES['KURTOSIS'] = [True, False]
FEATURES['VARIANCE'] = [True, False]
FEATURES['MODE'] = [True, False]


class Individual(object):
  
    def __init__(self, genome=None):
        if not genome:
            self.__genome = dict()
            self.__genome['COLOR'] = rd.choice(FEATURES['COLOR'])
            self.__genome['HISTOGRAM'] = rd.choice(FEATURES['HISTOGRAM'])
            self.__genome['DOG'] = rd.choice(FEATURES['DOG'])
            self.__genome['SIGMA1'] = rd.choice(FEATURES['SIGMA1'])
            self.__genome['SIGMA2'] = rd.choice(FEATURES['SIGMA2'])
            self.__genome['WAVELET'] = rd.choice(FEATURES['WAVELET'])
            self.__genome['WAVELET_REPEATS'] = rd.choice(FEATURES['WAVELET_REPEATS'])
            self.__genome['HORIZONTAL'] = rd.choice(FEATURES['HORIZONTAL'])
            self.__genome['VERTICAL'] = rd.choice(FEATURES['VERTICAL'])
            self.__genome['DIAGONAL'] = rd.choice(FEATURES['DIAGONAL'])
            self.__genome['APPROXIMATION'] = rd.choice(FEATURES['APPROXIMATION'])
            self.__genome['ENERGY'] = rd.choice(FEATURES['ENERGY'])
            # self.__genome['ENTROPY'] = rd.choice(FEATURES['ENTROPY'])
            # self.__genome['KURTOSIS'] = rd.choice(FEATURES['KURTOSIS'])
            self.__genome['MEAN'] = rd.choice(FEATURES['MEAN'])
            self.__genome['MEDIAN'] = rd.choice(FEATURES['MEDIAN'])
            # self.__genome['MODE'] = rd.choice(FEATURES['MODE'])
            self.__genome['VARIANCE'] = rd.choice(FEATURES['VARIANCE'])
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

    def mutate(self, tax):
        for feature in self.__genome:
            if rd.random() < tax:
                self.genome[feature] = rd.choice(FEATURES[feature])

    def to_json(self):
        genome = self.__genome
        score = self.__score
        return{
            'genome': genome,
            'score': score
        }

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score
