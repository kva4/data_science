import numpy as np


class BaseModel:
    def __init__(self, n):
        '''
            n - number of elements in model
        '''
        self.n = n
        self.__base_key = 'base'
        self.__x = None
        self.__y = {self.__base_key: None}

    def get_x(self):
        if self.__x is None:
            self.__x = [*range(self.n)]
        return self.__x

    def get_y(self, key=None):
        '''
            The method returns the model values.

            @param key - key of values in the model to return. If None, pure model values are returned.
        '''

        if key is not None and key != self.__base_key:
            if self.__y[key] is None:
                raise Exception(f'Noise with key: {key} has not been added.')
            return self.__y[key]

        if self.__y[self.__base_key] is None:
            self.__y[self.__base_key] = np.zeros(self.n)
            for x in self.get_x():
                self.__y[self.__base_key][x] = self._model(x)

        return self.__y[self.__base_key]

    @property
    def title(self):
        return self.__class__.__name__

    def add_noise(self, noise, noise_key, key=None):
        '''
            The method adds noise to the model.

            @param noise - noise values
            @param noise_key - key of the noise to store seperately from the model values or other noises
            @param key - key of values in the model to add the noise to
        '''

        if noise_key is None:
            raise Exception(f'Key: {noise_key} is None.')

        if noise is None:
            raise Exception('The noise value is None.')

        if len(noise) != self.n:
            raise Exception(
                f'Noise length: {len(noise)} is not equal to number of elements in model: {self.n}.')

        if key is not None and self.__y[key] is None:
            raise Exception(
                f'Noise with key: {key} has not been created.')

        base_key = key or self.__base_key

        self.__y[noise_key] = np.zeros(self.n)

        for i in range(self.n):
            self.__y[noise_key][i] = self.get_y(base_key)[i]+noise[i]

    def _model(self, x):
        return x
