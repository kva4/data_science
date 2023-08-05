import numpy as np
from .base_rvd import BaseRVD


class ExponentialRVD(BaseRVD):
    def __init__(self, n, alfa=None):
        '''
            n - об'єм вибірки
            alfa - параметр експоненційного закону розподілу ВВ
        '''
        BaseRVD.__init__(self, n)
        self.alfa = alfa or 1

    # ------------------------- експоненційний закон розводілу ВВ ----------------------------
    def _distribution(self):
        return np.random.exponential(self.alfa, self.n)
