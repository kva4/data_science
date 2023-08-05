from .base_model import BaseModel

'''
    a model with known arguments values and function values of the arguments
'''


class SimpleModel(BaseModel):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise Exception(
                'The number of arguments and values of the function must be equal.')

        super().__init__(n=len(x), args=x)
        self.data = dict(zip(x, y))

    def _model(self, x):
        return self.data[x]
