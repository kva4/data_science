import numpy as np


class LeastSquares(object):

    # ------------- МНК згладжуваннядля визначення стат. характеристик -------------
    @staticmethod
    def non_liner_fit(data_set, lstsq_range=3, print_stats=False):
        iter = len(data_set)
        Yin = np.asmatrix(data_set).T  # формування матриці вхідних даних
        F = np.ones((iter, lstsq_range))
        power_range = range(0, F.shape[1])
        for i in range(iter):  # формування структури вхідних матриць МНК
            F[i] = list(map(lambda x: float(i ** x), power_range))
        FT = F.T
        FFT = FT.dot(F)
        FFTI = np.linalg.inv(FFT)
        FFTIFT = FFTI.dot(FT)
        C = FFTIFT.dot(Yin)
        Yout = F.dot(C)

        if print_stats:
            print('Регресійна модель:')
            print(f'y(t) = {C[0, 0]} + {C[1, 0]} * t + {C[2, 0]} * t^2')

        return np.reshape(Yout.tolist(), iter)

    @staticmethod
    def non_liner_coef_fit(data_set, lstsq_range=3, print_stats=False):
        iter = len(data_set)
        Yin = np.asmatrix(data_set).T  # формування матриці вхідних даних
        F = np.ones((iter, lstsq_range))
        power_range = range(0, F.shape[1])
        for i in range(iter):  # формування структури вхідних матриць МНК
            F[i] = list(map(lambda x: float(i ** x), power_range))
        FT = F.T
        FFT = FT.dot(F)
        FFTI = np.linalg.inv(FFT)
        FFTIFT = FFTI.dot(FT)
        C = FFTIFT.dot(Yin)

        if print_stats:
            print('Регресійна модель:')
            print(f'y(t) = {C[0, 0]} + {C[1, 0]} * t + {C[2, 0]} * t^2')

        return np.reshape(C.tolist(), lstsq_range)

    @staticmethod
    # ---------------------------  МНК ПРОГНОЗУВАННЯ -------------------------------
    def non_liner_extrapol(S0, koef, lstsq_range=3):
        iter = len(S0)
        Yout_Extrapol = np.zeros(iter + koef)
        C = LeastSquares.non_liner_coef_fit(S0, lstsq_range, print_stats=True)
        for i in range(iter + koef):
            # проліноміальна крива МНК - прогнозування
            Yout_Extrapol[i] = C[0] + C[1] * i + (C[2] * i * i)

        return Yout_Extrapol
