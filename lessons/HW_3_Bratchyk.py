# --------------------------- Homework_3  ------------------------------------

'''

Виконав: ____
Homework_3, 2 рівень складності:
Умови apha beta gamma


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

        filter = ABF()
        estimates = list(map(lambda y: filter.predict(y), model.get_y('noise_av')))
        Yout_SV_AV_Detect_sliding_wind = np.array(estimates)
        Estimation.lstsq_estimation_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,
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
        Yout_SV_AV_Detect_sliding_wind = np.array(estimates)
        Estimation.lstsq_estimation_out(model.get_y('noise_av'), Yout_SV_AV_Detect_sliding_wind,
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

1. Для реальних даних - alpha-beta filter.
------------ ABF згладжена, Реальні дані очищена від АВ алгоритм sliding_wind -------------
кількість елементів ивбірки= 570
матиматичне сподівання ВВ= -0.9166920848587159
дисперсія ВВ = 56.11111203114723
СКВ ВВ= 7.490735079493016
Динамічна похибка моделі= 6.2413854636857575
-----------------------------------------------------
------------ ABF_модель_згладжування -------------
кількість елементів вбірки= 570
Коефіцієнт детермінації (ймовірність апроксимації)= 0.9806604257437076

2. Для квадратичної моделі - alpha-beta-gamma filter.




Висновок
------------------------------------------------------------------------------------------
alpha beta gamma фільтр краще апроксимую загальну лінію тренду як для квадратичної моделі так і для реальних даних, ніж alpha-beta filter.
Для квадратичної моделі, тому що тренд квадратичний (є прискорення), що враховано в alpha beta gamma фільтрі.
Для реальних даних, тому що є різкі зміни в даних(не лінійні за часом), які можна також трактувати як прискорення, що враховано в alpha beta gamma фільтрі.
'''
