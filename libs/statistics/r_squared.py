# ----- Коефіцієнт детермінації - оцінювання якості моделі --------
class RSquared(object):

    @staticmethod
    def score(SL, Yout, Text):
        # статистичні характеристики вибірки з урахуванням тренду
        iter = len(Yout)
        numerator = 0
        denominator_1 = 0
        for i in range(iter):
            numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
            denominator_1 = denominator_1 + SL[i]
        denominator_2 = 0
        for i in range(iter):
            denominator_2 = denominator_2 + \
                (SL[i] - (denominator_1 / iter)) ** 2
        R2_score_our = 1 - (numerator / denominator_2)
        print('------------', Text, '-------------')
        print('кількість елементів вбірки=', iter)
        print('Коефіцієнт детермінації (ймовірність апроксимації)=', R2_score_our)

        return R2_score_our
