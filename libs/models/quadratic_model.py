# ------------------- модель ідеального тренду (постійний закон)  ------------------
from .base_model import BaseModel


class QuadraticModel(BaseModel):
    def __init__(self, n, a=None, b=None, c=None):
        '''
            n - number of elements in model
            a - coefficient
            b - coefficient
            c - coefficient
        '''
        BaseModel.__init__(self, n)
        self.a = a or 0.0000005
        self.b = b or 0
        self.c = c or 0

    def _model(self, x):
        '''
            x - argument of the function
        '''
        # квадратична модель реального процесу
        return (self.a*x*x+self.b*x+self.c)
