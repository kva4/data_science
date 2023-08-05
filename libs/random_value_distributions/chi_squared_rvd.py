import numpy as np
from .base_rvd import BaseRVD


class ChiSquared(BaseRVD):
    def __init__(self, n, k):
        BaseRVD.__init__(self, n)
        self.k = k

    # ------------------------- хі квадрат закон розводілу ВВ ----------------------------
    def _distribution(self):
        return np.random.chisquare(self.k, self.n)
