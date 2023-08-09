import numpy as np
from .filter_calculating_params_type import FilterCalculatingParamsType as FCPType


class AlphaBetaGammaFilter:
    def __init__(self, prms_calc_type: FCPType = FCPType.DynamicBase, alpha=1, beta=1, gamma=1, dt=1):
        self.prms_calc_type = prms_calc_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x_1 = None
        self.v_1 = None
        self.a_1 = None
        self.y = []
        self.x_in = []
        self.dt = dt
        self.__set_params()

    def predict(self, x):
        self.x_in.append(x)
        self.__set_params()

        if not self.y:
            self.y.append(x)
            self.x_1 = x
            self.v_1 = 0
            self.a_1 = 0
            return self.y[-1]

        y_n = self.x_1 + self.alpha * (x - self.x_1)
        self.y.append(y_n)

        self.a_1 = self.a_1 + self.gamma * (x - self.x_1) / self.dt**2
        self.v_1 = (self.v_1 + self.a_1*self.dt) + \
            self.beta * (x - self.x_1) / self.dt
        self.x_1 = y_n + self.v_1 * self.dt + self.a_1 * self.dt/2

        return y_n

    @property
    def estimates(self):
        return self.y

    @property
    def __n(self):
        return len(self.x_in) - 1

    def __set_params(self):
        match self.prms_calc_type:
            case FCPType.Static:
                if self.alpha is None or self.beta is None or self.gamma is None:
                    raise ValueError('alpha and beta must be defined')
            case FCPType.DynamicBase:
                if self.__n < 1:
                    self.alpha = 1
                    self.beta = 1
                    self.gamma = 1
                else:
                    self.alpha = (3*(3*self.__n**2-3*self.__n+2)) / \
                        (self.__n*(self.__n+1)*(self.__n+2))
                    self.beta = (18*(2*self.__n-1)) / \
                        (self.__n*(self.__n+1)*(self.__n+2))
                    self.gamma = 60 / \
                        (self.__n * (self.__n + 1) * (self.__n + 2))
            case FCPType.DynamicOptimal:
                if self.__n < 1:
                    self.alpha = 1
                    self.beta = 1
                    self.gamma = 1
                else:
                    sigma_v = np.var(self.x_in)
                    sigma_w = 2.0*self.dt
                    lamb = (sigma_w*self.dt**2.0) / sigma_v
                    b = lamb/2.0 - 3
                    c = lamb/2.0 + 3
                    d = - 1
                    p = c - b**2.0/3.0
                    q = 2.0*b**3.0/27.0 - b*c/3.0 + d
                    v = np.sqrt(q**2+4.0*p**3.0/27.0)
                    z = -np.sqrt(q+v/2.0)
                    s = z - p/3.0*z - b/3.0

                    self.alpha = 1 - s**2.0
                    self.beta = 2.0*(1-s)**2.0
                    self.gamma = self.beta**2.0 / (2.0*self.alpha)
