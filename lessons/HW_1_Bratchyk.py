# --------------------------- Homework_1  ------------------------------------

'''

Виконав: Братчик Олександр
Homework_1, варіант 2, І рівень складності: 2
- Закон зміни похибки – рівномірний, нормальний;
- Закон зміни досліджуваного процесу (тренду) – постійна, квадратичний;
- Комбінаторика похибка / тренд – довільна;
-Реальні дані – 3 показники;

'''

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from libs.models import LinerModel, QuadraticModel, SimpleModel
from libs.random_value_distributions.normal_rvd import NormalRVD
from libs.statistics import *
from libs.load_data import *
matplotlib.use('WebAgg')
# ------------------------ ФУНКЦІЯ парсингу реальних даних --------------------------


def file_parsing(URL, File_name, Data_name):
    '''

    :param URL:
    :param File_name:
    :param Data_name:
    :return:
    '''

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
        # for name, values in d[[Data_name]].iteritems(): # приклад оновлення версій pandas для директиви iteritems
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real

# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------


def Plot_AV(S0_L, SV_L, Text):
    '''

    :param S0_L:
    :param SV_L:
    :param Text:
    :return:
    '''

    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return


def print_stats(stats: ch, title):
    print('------------', title, '-------------')
    print('матиматичне сподівання ВВ=', stats.mS())
    print('дисперсія ВВ =', stats.dS())
    print('СКВ ВВ=', stats.scvS())
    print('-----------------------------------------------------')

# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ----------------------------------


if __name__ == '__main__':
    print('Оберіть джерело вхідних даних та подальші дії:')
    print('1 - модель')
    print('2 - реальні дані')
    Data_mode = int(input('mode:'))

    if (Data_mode == 1):
        # ------------------------------ сегмент констант ---------------------------------------
        n = 10000                       # кількість реалізацій ВВ - об'єм вибірки
        iter = int(n)
        Q_AV = 3                        # коефіцієнт переваги АВ
        nAVv = 10
        # кількість АВ у відсотках та абсолютних одиницях
        nAV = int((iter * nAVv) / 100)
        dm = 0
        dsig = 5                        # параметри нормального закону розподілу ВВ: середнє та СКВ

        # ------------------------------ сегмент даних -------------------------------------------
        # ------------ виклики функцій моделей: тренд, аномального та нормального шуму  ----------
        # model = LinerModel(n)               # модель ідеального тренду (лінійний закон)
        # модель ідеального тренду (квадратичний закон)
        model = QuadraticModel(n)

        rv = NormalRVD(n, dm, dsig)      # модель нормальних помилок
        # rv = ExponentialRVD(n) # модель експоненційних помилок

        rv.plot_hist()
        rv.print_stats()

        # ---------------- модель виміру і випадкова величина(шум) ---------------
        model.add_noise(rv.values, 'noise')  # модель тренда + BB

        # ----------------------------- Нормальні похибки -----------------------------------------
        Plot_AV(model.get_y(), model.get_y('noise'), f'{model.title} + {rv.title}')
        Estimation.lstsq_estimation(model.get_y('noise'), 'Вибірка + Норм. шум')

    if (Data_mode == 2):
        # -------------------------------- Реальні дані -------------------------------------------
        meteo_data = get_clean_data()
        # check for null values
        model_mean = SimpleModel(
            meteo_data['date'].tolist(), meteo_data['mean'].tolist())
        model_max = SimpleModel(
            meteo_data['date'].tolist(), meteo_data['max'].tolist())
        model_min = SimpleModel(
            meteo_data['date'].tolist(), meteo_data['min'].tolist())

        stats_max = ch(model_max.get_y())
        print(f'Max temp in Basel between {model_max.get_x()[0]} - {model_max.get_x()[-1]}')
        stats_max.print_stats()
        stats_min = ch(model_min.get_y())
        print(f'\n\nMin temp in Basel between {model_min.get_x()[0]} - {model_min.get_x()[-1]}')
        stats_min.print_stats()
        stats_mean = ch(model_mean.get_y())
        print(f'\n\nMean temp in Basel between {model_mean.get_x()[0]} - {model_mean.get_x()[-1]}')
        stats_mean.print_stats()

        plt.clf()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))
        # plt.plot( model_max.get_x(), model_max.get_y())
        # plt.plot( model_min.get_x(), model_min.get_y())
        plt.plot(model_mean.get_x(), model_mean.get_y())
        plt.ylabel("Temperature C in Basel")
        plt.show()


'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.

1. Задані характеристики вхідної вибірка:
часова надмірність даних із квадратичним законом;
статистичні характеристики:
    закон розподілу ВВ - нормальний
    n = 10000   # кількість реалізацій ВВ - об'єм вибірки
    dm = 0
    dsig = 5    # параметри нормального закону розподілу ВВ: середнє та СКВ

2. Визначені характеристики вхідної вибірки:
часова надмірність даних із квадратичним законом підтверджена графіком;
статистичні характеристики:
    закон розподілу ВВ - нормальний, підтверджено гістограмою;
    -----------------------------------------------------------------------
    -------статистичні характеристики NormalRVD -----
    матиматичне сподівання ВВ= 0.026504749711095763
    дисперсія ВВ = 25.109399610530655
    СКВ ВВ= 5.010928018893372
    --------------------------------------------------------------
    ------------ Вибірка + Норм. шум -------------
    кількість елементів вбірки= 10000
    матиматичне сподівання ВВ= -0.02295295819325424
    дисперсія ВВ = 25.10526134088035
    СКВ ВВ= 5.010515077402756
    -----------------------------------------------------

3. Висновок
Відповідність заданих та обрахованих числових характеристик статистичної вибірки доводять адекватність розрахунків.
Розроблений скрипт можна використовувати для визначення статистичних характеристик реальних даних.

'''
