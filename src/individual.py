import random as rd

class Individual(object):

    def __init__(self, genome=None):
        if not genome:
            self.__genome = dict()
            self.__genome['COLOR'] =  rd.random()  #rd.choice(FEATURES['COLOR'])
            self.__genome['WAVELET'] = rd.random() #rd.choice(FEATURES['WAVELET'])

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
        self.__image_table= None

    @property
    def genome(self):
        return self.__genome

    @property
    def score(self):
        return self.__score

    @property
    def image_table(self):
        return self.__image_table

    @image_table.setter
    def image_table(self, image_table):
        self.__image_table = image_table

    @score.setter
    def score(self, score):
        self.__score = score

    @genome.setter
    def genome(self, genome):
        self.__genome = genome

    def mutate(self, best, worst):
        self.__score = 0
        for feature in self.__genome:
            distance_from_worst = rd.random() * (worst.genome[feature] - self.genome[feature])
            distance_from_best = rd.random() * (best.genome[feature] - self.genome[feature])

            new_value = self.genome[feature] + distance_from_best - distance_from_worst
            # normalize value to interval of 0 and 0.999999
            self.genome[feature] = max(min(new_value, 0.9999999), 0)



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
