import numpy as np
from .filter_calculating_params_type import FilterCalculatingParamsType as FCPType


class AlphaBetaFilter:
    def __init__(self, prms_calc_type: FCPType = FCPType.DynamicBase, alpha=1, beta=1, dt=1):
        self.prms_calc_type = prms_calc_type
        self.alpha = alpha
        self.beta = beta
        self.x_1 = None
        self.v_1 = None
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
            return self.y[-1]

        self.y.append(self.x_1 + self.alpha * (x - self.x_1))
        self.v_1 = self.v_1 + self.beta * (x - self.x_1) / self.dt
        self.x_1 = self.y[-1] + self.v_1 * self.dt

        return self.y[-1]

    @property
    def estimates(self):
        return self.y

    @property
    def __n(self):
        return len(self.x_in) - 1

    def __set_params(self):
        match self.prms_calc_type:
            case FCPType.Static:
                if self.alpha is None or self.beta is None:
                    raise ValueError('alpha and beta must be defined')
            case FCPType.DynamicBase:
                if self.__n < 1:
                    self.alpha = 1
                    self.beta = 1
                else:
                    self.alpha = 2*(2*self.__n-1)/(self.__n*(self.__n+1))
                    self.beta = 6 / (self.__n * (self.__n + 1))
            case FCPType.DynamicOptimal:
                if self.__n < 1:
                    self.alpha = 1
                    self.beta = 1
                else:
                    sigma_v = np.var(self.x_in)
                    sigma_w = 2.0*self.dt
                    lamb = (sigma_w*self.dt**2.0) / sigma_v

                    self.alpha = 1/8.0*(-lamb**2.0 - 8.0*lamb +
                                        (lamb+4)*np.sqrt(lamb**2.0 + 8.0*lamb))
                    self.beta = 1/4.0*(lamb**2.0 + 4.0*lamb -
                                       lamb*np.sqrt(lamb**2.0 + 8.0*lamb))
