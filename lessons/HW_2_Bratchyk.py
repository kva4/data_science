# --------------------------- Homework_2  ------------------------------------

'''

Виконав: ____
Homework_2, варіант __, ___ рівень складності:
Умови _____

Виконання:
1. Відкинути зайве, додати необхідне;
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
import matplotlib.pyplot as plt
import pandas as pd
from libs.models.quadratic_model import QuadraticModel
from libs.random_value_distributions.normal_rvd import NormalRVD
from libs.anomaly_detection import Detection
from libs.regression_analysis import lstsq
from libs.statistics import r2, Estimation

# ------------------------ ФУНКЦІЯ парсингу реальних даних --------------------------

def file_parsing(URL, File_name, Data_name):
    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real


# ----- статистичні характеристики лінії тренда  --------
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
        nAV = int((iter * nAVv) / 100)  # кількість АВ у відсотках та абсолютних одиницях
        dm = 0
        dsig = 5  # параметри нормального закону розподілу ВВ: середне та СКВ

        # ------------------------------ сегмент даних ---------------------------
        # ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
        model = QuadraticModel(n)

        # ----------------------------- Нормальні похибки ------------------------------------

        rv = NormalRVD(n, dm, dsig)
        rv.plot_hist()
        model.add_noise(rv.values, 'noise')
        Plot_AV(model.get_y(), model.get_y('noise'), 'квадратична модель + Норм. шум')
        Estimation.lstsq_estimation(model.get_y('noise'), 'Вибірка + Норм. шум')

        # ----------------------------- Аномальні похибки ------------------------------------
        model.add_noise(rv.anomaly_values, 'noise_av')
        Plot_AV(model.get_y(), model.get_y('noise_av'), 'квадратична модель + Норм. шум + АВ')
        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка з АВ')

    if (Data_mode == 2):
        # SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'Купівля')  # реальні дані
        SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls',
                             'Продаж')  # реальні дані
        # SV_AV = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'КурсНбу')  # реальні дані

        S0 = SV_AV
        n = len(S0)
        iter = int(n)  # кількість реалізацій ВВ
        Plot_AV(SV_AV, SV_AV, 'Коливання курсу USD в 2022 році за даними Ощадбанк')
        Estimation.lstsq_estimation(SV_AV, 'Коливання курсу USD в 2022 році за даними Ощадбанк')

    if (Data_mode == 3):
        print('Бібліотеки Python для реалізації методів статистичного навчання:')
        print('https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html')
        print('https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html')
        print('https://scikit-learn.org/stable/modules/sgd.html#regression')
        sys.exit(0)

    # ------------------- вибір функціоналу статистичного навчання -----------------------

    print('Оберіть функціонал процесів навчання:')
    print('1 - детекція та очищення від АВ: метод medium')
    print('2 - детекція та очищення від АВ: метод MNK')
    print('3 - детекція та очищення від АВ: метод sliding window')
    print('4 - МНК згладжування')
    print('5 - МНК прогнозування')
    mode = int(input('mode:'))
    SV_AV = model.get_y('noise_av').copy()

    if (mode == 1):
        print('Вибірка очищена від АВ метод medium')
        # --------- Увага!!! якість результату залежить від якості еталонного вікна -----------
        N_Wind_Av = 5  # розмір ковзного вікна для виявлення АВ
        Q = 1.6  # коефіцієнт виявлення АВ

        clean_fn = lambda x: Detection.detect_medium(x, Q, N_Wind_Av)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від алгоритм medium АВ')
        Yout_SV_AV_Detect = lstsq.non_liner_fit(model.get_y('noise_av'))

        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect, 'МНК Вибірка відчищена від АВ алгоритм medium')
        Plot_AV(model.get_y(), model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм medium')

    if (mode == 2):
        print('Вибірка очищена від АВ метод MNK')
        # ------------------- Очищення від аномальних похибок МНК --------------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        Q_MNK = 7  # коефіцієнт виявлення АВ

        clean_fn = lambda x: Detection.detect_lstsq(x, Q_MNK, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм MNK')
        Yout_SV_AV_Detect_MNK = lstsq.non_liner_fit(model.get_y('noise_av'))

        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect_MNK, 'МНК Вибірка очищена від АВ алгоритм MNK')
        Plot_AV(model.get_y(), model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм MNK')

    if (mode == 3):
        print('Вибірка очищена від АВ метод sliding_wind')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')
        Yout_SV_AV_Detect_sliding_wind = lstsq.non_liner_fit(model.get_y('noise_av'))

        Stat_characteristics_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind, 'МНК Вибірка очищена від АВ алгоритм sliding_wind')
        Plot_AV(model.get_y(), model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')

    if (mode == 4):
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

    if (mode == 5):
        print('MNK ПРОГНОЗУВАННЯ')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        koef_Extrapol = 0.5  # коефіціент прогнозування: співвідношення інтервалу спостереження до  інтервалу прогнозування
        koef = mt.ceil(n * koef_Extrapol)  # інтервал прогнозу по кількісті вимірів статистичної вибірки

        clean_fn = lambda x: Detection.detect_sliding_wind(x, n_Wind)
        model.clean_noise(clean_fn, 'noise_av')

        Estimation.lstsq_estimation(model.get_y('noise_av'), 'Вибірка очищена від АВ алгоритм sliding_wind')
        Yout_SV_AV_Detect_sliding_wind = lstsq.non_liner_extrapol(model.get_y('noise_av'), koef)

        stats = Estimation.lstsq_estimation(Yout_SV_AV_Detect_sliding_wind, 'MNK ПРОГНОЗУВАННЯ, вибірка очищена від АВ алгоритм sliding_wind')
        print('Довірчий інтервал прогнозованих значень за СКВ=', stats.scvS*koef)

        Plot_AV(Yout_SV_AV_Detect_sliding_wind, model.get_y('noise_av'),
                'MNK ПРОГНОЗУВАННЯ: Вибірка очищена від АВ алгоритм sliding_wind')

'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.
------------------------------------------------------------------------------------------

Висновок
------------------------------------------------------------------------------------------

'''