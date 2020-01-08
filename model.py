import numpy as np


class Model(object):

    def train(self, data, labels):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError

    def error(self, X, y):
        total = y.size
        return np.sum(y == self.predict(X)) / total

    def visualize(self, outfile='', directory=''):
        raise NotImplementedError


class Data(object):

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def k_fold_partition(self, k):
        raise NotImplementedError

class Optimizer:

    def __init__(self, cost_function, method):
        self.cost_function = cost_function
        self.method = method
