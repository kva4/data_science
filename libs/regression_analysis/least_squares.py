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

        if print_stats:
            print('Регресійна модель:')
            print(f'y(t) = {C[0, 0]} + {C[1, 0]} * t + {C[2, 0]} * t^2')

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

        return C

    @staticmethod
    # ---------------------------  МНК ПРОГНОЗУВАННЯ -------------------------------
    def non_liner_extrapol(S0, koef):
        iter = len(S0)
        Yout_Extrapol = np.zeros((iter + koef, 1))
        C = LeastSquares.non_liner_coef_fit(S0, print_stats=True)
        for i in range(iter + koef):
            Yout_Extrapol[i, 0] = C[0, 0] + C[1, 0] * i + (C[2, 0] * i * i)  # проліноміальна крива МНК - прогнозування

        return Yout_Extrapol