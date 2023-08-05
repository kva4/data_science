import numpy as np
from .base_rvd import BaseRVD


class UniformRVD(BaseRVD):
    def __init__(self, n, a, b):
        BaseRVD.__init__(self, n)
        self.a = a
        self.b = b

    # ------------------------- рівномірний закон розводілу ВВ ----------------------------
    def _distribution(self):
        values = np.zeros(self.n)
        for i in range(self.n):
            values[i] = np.random.uniform(self.a, self.b)

        return values
