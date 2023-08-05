import numpy as np
import matplotlib.pyplot as plt
from ..statistics.characteristics import Characteristics
import matplotlib
import itertools
matplotlib.use('WebAgg')


class BaseRVD:
    def __init__(self, n, n_av=None) -> None:
        '''
            :param n: обєм вибірки
            :param n_av: кількість АВ у відсотках та абсолютних одиницях
        '''
        self.n = n
        self._values = None
        self._values_av = None
        # default value: 10% від обєму вибірки
        self.n_av = n_av or int((self.n * 10) / 100)

    @property
    def values(self):
        if self._values is None:
            self._values = self._distribution()
        return self._values

    @property
    def anomaly_values(self):
        if self._values_av is None:
            self._values_av = self.values.copy()
            rnd_indx = np.random.randint(0, self.n, self.n_av)
            av_dist = self._distribution_av()
            list(map(lambda i, v: self._values_av.__setitem__(i, v), rnd_indx, av_dist))

        return self._values_av

    @property
    def title(self):
        return self.__class__.__name__

    def print_stats(self):
        print(f'-------статистичні характеристики {self.title} -----')
        Characteristics(self.values).print_stats()
        print('--------------------------------------------------------------')

    def plot_hist(self):
        # гістограма закону розподілу ВВ
        plt.hist(self.values, bins=20, facecolor="blue", alpha=0.5)
        plt.show()

    def _distribution(self):
        return np.zeros(self.n)

    def _distribution_av(self):
        return np.zeros(self.n)
