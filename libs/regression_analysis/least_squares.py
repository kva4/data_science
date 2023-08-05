import numpy as np


class LeastSquares(object):

    # ------------- МНК згладжуваннядля визначення стат. характеристик -------------
    @staticmethod
    def non_liner_fit(data_set, print_stats=False):
        iter = len(data_set)
        Yin = np.zeros((iter, 1))
        F = np.ones((iter, 3))
        for i in range(iter):  # формування структури вхідних матриць МНК
            Yin[i, 0] = float(data_set[i])  # формування матриці вхідних даних
            F[i, 1] = float(i)
            F[i, 2] = float(i * i)
        FT = F.T
        FFT = FT.dot(F)
        FFTI = np.linalg.inv(FFT)
        FFTIFT = FFTI.dot(FT)
        C = FFTIFT.dot(Yin)
        Yout = F.dot(C)

        return Yout

    @staticmethod
    def non_liner_coef_fit(data_set, print_stats=False):
        iter = len(data_set)
        Yin = np.zeros((iter, 1))
        F = np.ones((iter, 3))
        for i in range(iter):  # формування структури вхідних матриць МНК
            Yin[i, 0] = float(data_set[i])  # формування матриці вхідних даних
            F[i, 1] = float(i)
            F[i, 2] = float(i * i)
        FT = F.T
        FFT = FT.dot(F)
        FFTI = np.linalg.inv(FFT)
        FFTIFT = FFTI.dot(FT)
        C = FFTIFT.dot(Yin)

        if print_stats:
            print('Регресійна модель:')
            print(f'y(t) = {C[0, 0]} + {C[1, 0]} * t + {C[2, 0]} * t^2')

        return C[1, 0]
