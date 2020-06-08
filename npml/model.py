import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Model(object):

    def train(self, inputs, target):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def error(self, inputs, target):
        total = target.size
        return np.sum(target == self.predict(inputs)) / total

    def visualize(self, outfile='', directory=''):
        raise NotImplementedError


class Loss(object):

    def __call__(self, outputs, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Data(object):

    def __init__(self, inputs, target, normalize=False):
        self.inputs = inputs
        self.target= target
        self.normalize = normalize
        if normalize:
            self.inputs = Data.normalize(self.inputs)

    def k_fold_partition(self, k):
        raise NotImplementedError

    def build_batch(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=(batch_size))
        inputs = np.take(self.inputs, indices, axis=0)
        if self.normalize:
            inputs = Data.normalize(inputs)
        return inputs, np.take(self.target, indices, axis=0)

    def __len__(self):
        return len(self.target)

    @staticmethod
    def normalize(data, ax=0):
        mean = np.mean(data, axis=ax)
        std = np.std(data, axis=ax)
        data = (data - mean) / std
        return data

class Optimizer(object):

    def __init__(self, model, learning_rate, cost_function):
        pass

    def step():
        pass
