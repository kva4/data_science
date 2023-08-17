from ..regression_analysis import lstsq
from .characteristics import Characteristics as ch
import numpy as np
import math as mt


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
            estimation_set[i] = data_set[i] - y_est[i]

        stats = ch(estimation_set)

        print('------------', title, '-------------')
        print('кількість елементів вбірки=', n)
        stats.print_stats()
        print('-----------------------------------------------------')
        return stats

    @staticmethod
    def lstsq_estimation_out(SL_in, SL, title=''):
        # статистичні характеристики вибірки з урахуванням тренду
        Yout = lstsq.non_liner_fit(SL)
        iter = len(Yout)
        SL0 = np.zeros((iter))
        for i in range(iter):
            SL0[i] = SL[i] - Yout[i]
        mS = np.median(SL0)
        dS = np.var(SL0)
        scvS = mt.sqrt(dS)
        # глобальне лінійне відхилення оцінки - динамічна похибка моделі
        Delta = 0
        for i in range(iter):
            Delta = Delta + abs(SL_in[i] - Yout[i])
        Delta_average_Out = Delta / (iter + 1)
        print('------------', title, '-------------')
        print('кількість елементів ивбірки=', iter)
        print('матиматичне сподівання ВВ=', mS)
        print('дисперсія ВВ =', dS)
        print('СКВ ВВ=', scvS)
        print('Динамічна похибка моделі=', Delta_average_Out)
        print('-----------------------------------------------------')
        return
