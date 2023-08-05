import numpy as np
import math as mt
from ..regression_analysis.least_squares import LeastSquares as lstsq


class Detection(object):

    @staticmethod
    def detect_lstsq(data_set, Q, n_Wind):
        # ---- параметри циклів ----
        iter = len(data_set)
        fixed_data_set = data_set.copy()
        j_Wind = mt.ceil(iter - n_Wind) + 1
        S0_Wind = np.zeros((n_Wind))

        # -------- еталон  ---------
        Speed_standart = lstsq.non_liner_coef_fit(data_set)
        Yout_S0 = lstsq.non_liner_fit(data_set)

        # ---- ковзне вікно ---------
        for j in range(j_Wind):
            for i in range(n_Wind):
                l = (j + i)
                S0_Wind[i] = data_set[l]
            # - Стат хар ковзного вікна --
            dS = np.var(S0_Wind)
            scvS = mt.sqrt(dS)
            # --- детекція та заміна АВ --
            Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
            Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
            if Speed_1 > Speed_standart_1:
                # детектор виявлення АВ
                fixed_data_set[l] = Yout_S0[l, 0]
        return fixed_data_set

    @staticmethod
    def detect_sliding_wind(data_set, n_Wind):
        # ---- параметри циклів ----
        iter = len(data_set)
        j_Wind = mt.ceil(iter - n_Wind) + 1
        S0_Wind = np.zeros((n_Wind))
        Midi = np.zeros((iter))
        # ---- ковзне вікно ---------
        for j in range(j_Wind):
            for i in range(n_Wind):
                l = (j + i)
                S0_Wind[i] = data_set[l]
            # - Стат хар ковзного вікна --
            Midi[l] = np.median(S0_Wind)
        # ---- очищена вибірка  -----
        S0_Midi = np.zeros((iter))
        for j in range(iter):
            S0_Midi[j] = Midi[j]
        for j in range(n_Wind):
            S0_Midi[j] = data_set[j]
        return S0_Midi

    # ------------------------------ Виявлення АВ за алгоритмом medium -------------------------------------
    @staticmethod
    def Sliding_Window_AV_Detect_medium(data_set, n_Wind, Q):
        # ---- параметри циклів ----
        iter = len(data_set)
        fixed_data_set = data_set.copy()
        j_Wind = mt.ceil(iter - n_Wind) + 1
        S0_Wind = np.zeros((n_Wind))
        # -------- еталон  ---------
        j = 0
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = data_set[l]
            dS_standart = np.var(S0_Wind)
            scvS_standart = mt.sqrt(dS_standart)
        # ---- ковзне вікно ---------
        for j in range(j_Wind):
            for i in range(n_Wind):
                l = (j + i)
                S0_Wind[i] = data_set[l]
            # - Стат хар ковзного вікна --
            mS = np.median(S0_Wind)
            dS = np.var(S0_Wind)
            scvS = mt.sqrt(dS)
            # --- детекція та заміна АВ --
            if scvS > (Q * scvS_standart):
                # детектор виявлення АВ
                fixed_data_set[l] = mS
        return fixed_data_set
