import numpy as np
import math as mt


class Characteristics:
    def __init__(self, values):
        self.values = values

    def expected_value(self):
        return np.median(self.values)

    def variance(self):
        return np.var(self.values)

    def standard_deviation(self):
        return mt.sqrt(self.variance())

    @property
    def mS(self):
        return self.expected_value()

    @property
    def dS(self):
        return self.variance()

    @property
    def scvS(self):
        return self.standard_deviation()

    def print_stats(self):
        print('матиматичне сподівання ВВ=', self.mS)
        print('дисперсія ВВ =', self.dS)
        print('СКВ ВВ=', self.scvS)
