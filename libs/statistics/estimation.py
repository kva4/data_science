from ..regression_analysis import lstsq
from .characteristics import Characteristics as ch
import numpy as np


class Estimation(object):

    @staticmethod
    def lstsq_estimation(data_set, title=''):
        '''
            The method estimates the statistical characteristics of the data set using the least squares method.
            Best fit if the stistic characteristics are close to 0.

            @param data_set - data set
            @param print_stats - print statistics flag
        '''

        y_est = lstsq.non_liner_fit(data_set)
        n = len(y_est)
        estimation_set = np.zeros((n))
        for i in range(n):
            estimation_set[i] = data_set[i] - y_est[i, 0]

        stats = ch(estimation_set)

        print('------------', title, '-------------')
        print('кількість елементів вбірки=', n)
        stats.print_stats()
        print('-----------------------------------------------------')
        return stats
