# ------------------- модель ідеального тренду (квадратичний закон)  ------------------
from .base_model import BaseModel


class LinerModel(BaseModel):
    def __init__(self, n, a=None, b=None):
        '''
            a - slope of the line
            b - the intercept
        '''
        BaseModel.__init__(self, n)
        self.n = n
        self.a = a or 0.0005
        self.b = b or (-5)
        self.model = None

    def _model(self, x):
        '''
            x - argument of the function
        '''

        return (x*self.a + self.b)    # лінійна модель реального процесу