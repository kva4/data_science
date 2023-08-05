import numpy as np
from .base_rvd import BaseRVD


class NormalRVD(BaseRVD):
    def __init__(self, n, dm, dsig, q_av=None, n_av=None):
        '''
            :param dm:
            :param dsig: стандартне відхилення
            :param q_av: коефіцієнт переваги АВ
        '''

        BaseRVD.__init__(self, n, n_av)
        self.dm = dm
        self.dsig = dsig
        self.q_av = q_av or 3

    # ------------------------- нормальний закон розводілу ВВ ----------------------------
    def _distribution(self):
        # нормальний закон розподілу ВВ з вибіркою обємом iter та параметрами: dm, dsig
        return np.random.normal(self.dm, self.dsig, self.n)

    def _distribution_av(self):
        return np.random.normal(self.dm, (self.q_av * self.dsig), self.n_av)
