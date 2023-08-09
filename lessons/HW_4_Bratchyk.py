# --------------------------- Homework_3  ------------------------------------

'''

Виконав: ____
Homework_4, ___ рівень складності:
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
import time
import numpy as np
import math as mt
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# ----- статистичні характеристики лінії тренда  --------
def Stat_characteristics_out_expo (SL_in, SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = lstsq.non_liner_fit(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter ))
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
    print('------------', Text ,'-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Динамічна похибка моделі=', Delta_average_Out)
    print('-----------------------------------------------------')
    return

def exp_with_lstsq(data_set, print_stats=False):
    C = lstsq.non_liner_coef_fit(data_set, lstsq_range=4, print_stats=print_stats)
    c0=C[0, 0]
    c1=C[1, 0]
    c2=C[2, 0]
    c3=C[3, 0]
    a3 = 3 * (c3 / c2)
    a2 = (2*c2)/(a3**2)
    a0 = c0 - a2
    a1 = c1-(a2*a3)
    print('Регресійна модель:')
    print('y(t) = ', a0, ' + ', a1, ' * t', ' + ', a2, ' * exp(', a3, ' * t )')
    y_out = np.zeros((len(data_set), 1))
    for i in range(iter):
        y_out[i, 0]=a0 + a1 * i + a2 * mt.exp(a3 * i)
    return y_out

# -------------------------------- Expo_scipy ---------------------------------
def Expo_Regres (Yin, bstart):
    def func_exp(x, a, b, c, d):
        print('Регресійна модель:')
        print('y(t) = ', c, ' + ', d, ' * t', ' + ', a, ' * exp(', b, ' * t)')
        return a * np.exp(b * x) + c + (d * x)
    # ------ эмпирические коэффициенты старта для bstart=1202.059798705
    aStart=bstart/10
    bStart=bstart/1000
    cStart=bstart+10
    dStart=bstart/10
    iter = len(Yin)
    x_data = np.ones((iter))
    y_data = np.ones((iter))
    for i in range(iter):
        x_data[i] = i
        y_data[i] = Yin[i]
    # popt, pcov = curve_fit(func_exp, x_data, y_data, p0=(12, 0.0012, 1200, 120))
    popt, pcov = curve_fit(func_exp, x_data, y_data, p0=(aStart, bStart, cStart, dStart))

    return func_exp(x_data, *popt)

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
        # ------------------------------ сегмент констант ---------------------------
    n = 10000
    iter = int(n)  # кількість реалізацій ВВ
    Q_AV = 3  # коефіцієнт переваги АВ
    nAVv = 10
    nAV = int((iter * nAVv) / 100)  # кількість АВ у відсотках та абсолютних одиницях
    dm = 0
    dsig = 5  # параметри нормального закону розподілу ВВ: середне та СКВ
    model = QuadraticModel(n)
    rv = NormalRVD(n, dm, dsig)
    model.add_noise(rv.values, 'noise')
    model.add_noise(rv.anomaly_values, 'noise_av')

    # ------------------- вибір функціоналу статистичного навчання моделі-----------------------
    print('Оберіть функціонал процесів навчання моделі:')
    print('1 - МНК згладжування')
    print('2 - МНК прогнозування')
    print('3 - МНК експонента за R&D')
    print('4 - Експонента за класикою')
    mode = int(input('mode:'))

    if (mode == 1):
        print('MNK згладжена вибірка очищена від АВ алгоритм sliding_wind')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')
        Yout_SV_AV_Detect_sliding_wind = lstsq.non_liner_fit(model.get_y('noise_av'))
        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind, 'MNK згладжена, вибірка очищена від АВ алгоритм sliding_wind')

        # --------------- Оцінювання якості моделі та візуалізація -------------------------
        r2.score(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind, 'MNK_модель_згладжування')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'), 'MNK Вибірка очищена від АВ алгоритм sliding_wind')

    if (mode == 2):
        print('MNK ПРОГНОЗУВАННЯ')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        koef = mt.ceil(len(model.get_x()) * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки

        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')


        Yout_SV_AV_Detect_sliding_wind = lstsq.non_liner_extrapol(model.get_y('noise_av'), koef)

        stats = Estimation.lstsq_estimation(Yout_SV_AV_Detect_sliding_wind, 'MNK ПРОГНОЗУВАННЯ, вибірка очищена від АВ алгоритм sliding_wind')
        print('Довірчий інтервал прогнозованих значень за СКВ=', stats.scvS*koef)

        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'),
                'MNK ПРОГНОЗУВАННЯ: Вибірка очищена від АВ алгоритм sliding_wind')

    if (mode == 3):
        print('MNK ЕКСПОНЕНТА')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')
        StartTime = time.time()  # фіксація часу початку обчислень
        Yout_SV_AV_Detect_sliding_wind = exp_with_lstsq(model.get_y('noise_av'))
        totalTime = (time.time() - StartTime)  # фіксація часу, на очищення від АВ
        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,
                             'MNK ЕКСПОНЕНТА, вибірка очищена від АВ алгоритм sliding_wind')

        # --------------- Оцінювання якості моделі та візуалізація -------------------------
        r2.score(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,  'MNK ЕКСПОНЕНТА_модель_згладжування')
        print('totalTime =', totalTime, 's')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'),
                'MNK ЕКСПОНЕНТА: Вибірка очищена від АВ алгоритм sliding_wind')

    if (mode == 4):
        # TODO: fix data for Stat_characteristics_out_expo & r2.score
        print('Регресія ЕКСПОНЕНТА')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки

        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')


        StartTime = time.time()  # фіксація часу початку обчислень
        Yout_SV_AV_Detect_sliding_wind = Expo_Regres(model.get_y('noise_av'), 10)
        totalTime = (time.time() - StartTime)  # фіксація часу, на очищення від АВ
        Stat_characteristics_out_expo(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,
                             'Регресія ЕКСПОНЕНТА, вибірка очищена від АВ алгоритм sliding_wind')

        # --------------- Оцінювання якості моделі та візуалізація -------------------------
        # r2.score(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind, 'Регресія ЕКСПОНЕНТА_модель_згладжування')
        print('totalTime =', totalTime, 's')
        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'),
                'Регресія ЕКСПОНЕНТА: Вибірка очищена від АВ алгоритм sliding_wind')



'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.
------------------------------------------------------------------------------------------

Висновок
------------------------------------------------------------------------------------------

'''
