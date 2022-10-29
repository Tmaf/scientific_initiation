import random as rd
import pywt

FEATURES = dict()

FEATURES['COLOR'] = ['r', 'g', 'b', 'h', 's', 'v', 'y']
FEATURES['WAVELET'] = pywt.wavelist(kind="discrete")


class Individual(object):

    def __init__(self, genome=None):
        if not genome:
            self.__genome = dict()
            self.__genome['COLOR'] = rd.choice(FEATURES['COLOR'])
            self.__genome['WAVELET'] = rd.choice(FEATURES['WAVELET'])

            self.__genome['SIGMA1'] = rd.random()
            self.__genome['SIGMA2'] = rd.random()
            self.__genome['WAVELET_REPEATS'] = rd.random()
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
            if feature not in ['COLOR', 'WAVELET']:

                worst_variation = rd.random() * (worst.genome[feature] - self.genome[feature])
                best_variation = rd.random() * (best.genome[feature] - self.genome[feature])

                new_value = self.genome[feature] + best_variation - worst_variation
                # normalize value to interval of 0 and 1
                self.genome[feature] = max(min(new_value, 1), 0)

            else:
                if rd.random() < tax:
                    if rd.random() < tax:
                        self.genome[feature] = rd.choice(FEATURES[feature])
                    else:
                        self.genome[feature] = best.genome[feature]

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
