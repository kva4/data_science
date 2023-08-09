# --------------------------- Homework_3  ------------------------------------

'''

Виконав: ____
Homework_3, ___ рівень складності:
Умови _____

Виконання:
1. Обрати рівень складності, відкинути зайве, додати необхідне у прикладі;
2. Написати власний скрипт.

Package                      Version
---------------------------- -----------

pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
xlrd                         2.0.1
matplotlib                   3.6.2

'''


import sys
import numpy as np
import math as mt
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from libs.models import QuadraticModel, SimpleModel
from libs.random_value_distributions.normal_rvd import NormalRVD
from libs.anomaly_detection import Detection
from libs.regression_analysis import lstsq
from libs.statistics import r2, Estimation
from libs.load_data import *
from libs.estimators import *


def Stat_characteristics_out(SL_in, SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = lstsq.non_liner_fit(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    # глобальне лінійне відхилення оцінки - динамічна похибка моделі
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL_in[i] - Yout[i, 0])
    Delta_average_Out = Delta / (iter + 1)
    print('------------', Text, '-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Динамічна похибка моделі=', Delta_average_Out)
    print('-----------------------------------------------------')
    return

# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------
def Plot_AV(S0_L, SV_L, Text):
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return

# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ------------------------------

if __name__ == '__main__':

    # ------------------------------ Джерело вхідних даних ---------------------------

    print('Оберіть джерело вхідних даних та подальші дії:')
    print('1 - модель')
    print('2 - реальні дані')
    print('3 - Бібліотеки для статистичного навчання -->>> STOP')
    Data_mode = int(input('mode:'))

    if (Data_mode == 1):
        # ------------------------------ сегмент констант ---------------------------
        n = 10000
        iter = int(n)  # кількість реалізацій ВВ
        Q_AV = 3  # коефіцієнт переваги АВ
        nAVv = 10
        # кількість АВ у відсотках та абсолютних одиницях
        nAV = int((iter * nAVv) / 100)
        dm = 0
        dsig = 5  # параметри нормального закону розподілу ВВ: середне та СКВ

        # ------------------------------ сегмент даних ---------------------------
        # ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
        model = QuadraticModel(n)

        # ----------------------------- Нормальні похибки ------------------------------------

        rv = NormalRVD(n, dm, dsig)
        rv.plot_hist()
        model.add_noise(rv.values, 'noise')
        Plot_AV(model.get_y(), model.get_y('noise'),
                'квадратична модель + Норм. шум')
        Estimation.lstsq_estimation(
            model.get_y('noise'), 'Вибірка + Норм. шум')

        # ----------------------------- Аномальні похибки ------------------------------------
        model.add_noise(rv.anomaly_values, 'noise_av')
        Plot_AV(model.get_y(), model.get_y('noise_av'),
                'квадратична модель + Норм. шум + АВ')
        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка з АВ')

        print('ABF згладжена вибірка очищена від АВ алгоритм sliding_wind')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y(
            'noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')

        filter = ABGF()
        estimates = list(map(lambda y: filter.predict(y), model.get_y('noise_av')))
        Yout_SV_AV_Detect_sliding_wind = np.array(estimates).reshape(-1, 1)
        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,
                                 'ABF згладжена, вибірка очищена від АВ алгоритм sliding_wind')

        # --------------- Оцінювання якості моделі та візуалізація -------------------------
        r2.score(model.get_y('noise_av'),
                 Yout_SV_AV_Detect_sliding_wind, 'ABF_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'),
                'ABF Вибірка очищена від АВ алгоритм sliding_wind')

    if (Data_mode == 2):
        # -------------------------------- Реальні дані -------------------------------------------
        meteo_data = get_clean_data()
        model = SimpleModel(meteo_data['date'].tolist(), meteo_data['mean'].tolist())
        model.copy_base_to('noise_av')

        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Реальні дані')
        Plot_AV(model.get_y(), model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм medium')

        #filter = ABF(prms_calc_type=FCPType.DynamicOptimal)
        filter = ABGF(prms_calc_type=FCPType.DynamicOptimal)
        estimates = list(map(lambda y: filter.predict(y), model.get_y('noise_av')))
        Yout_SV_AV_Detect_sliding_wind = np.array(estimates).reshape(-1, 1)
        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,
                                 'ABF згладжена, Реальні дані очищена від АВ алгоритм sliding_wind')

        # --------------- Оцінювання якості моделі та візуалізація -------------------------
        r2.score(model.get_y('noise_av'),
                 Yout_SV_AV_Detect_sliding_wind, 'ABF_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'),
                'ABF Вибірка очищена від АВ алгоритм sliding_wind')

    if (Data_mode == 3):
        print('Бібліотеки Python для реалізації методів статистичного навчання:')
        print('https://filterpy.readthedocs.io/en/latest/index.html#')
        print('https://unit8co.github.io/darts/generated_api/darts.models.filtering.kalman_filter.html')
        sys.exit(0)


'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.
------------------------------------------------------------------------------------------

Висновок
------------------------------------------------------------------------------------------

'''
