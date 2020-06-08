from model import *
import numpy as np
from utilities import *

class MSELoss(Loss):

    def __call__(self, output, labels):
        self.cache = (output, labels)
        return np.square(np.subtract(output, labels)).mean()

    def backward(self):
        output, labels = self.cache
        n = output.shape[0]
        dloss = 2 * (output - labels) / n
        return dloss


class CrossEntropyLoss(Loss):

    def __call__(self, output, labels):
        #NOTE:Add assertion to verify outputs form a probability
        self.cache = (output, labels)
        labels = np.bincount(labels) // len(labels)
        return -labels * np.log2(output) #NOTE:NEED to divide by N or just take mean

    def backward(self):
        raise NotImplementedError


class MAELoss(Loss):

    def __call__(self, output, labels):
        self.cache = (output, labels)
        return np.absolute(np.subtract(output, labels)).mean()

    def backward(self):
        output, labels = self.cache
        signs = np.sign(output)
        loc = np.where(signs==0)
        signs[loc] = np.random.uniform(low=-1, high=1, size=loc.shape)
        return signs


class SoftmaxLoss(Loss):

    #NOTE: softmax is not an actual loss its really softmax activation + cross entropy loss
    #Expects 0 1 2 3 4 etc classes
    def __call__(self, output, labels):
        self.cache = (output, labels)
        n = len(labels)
        output = softmax(output)
        output = output[np.arange(n), labels]
        return np.mean(-np.log2(output))

    def backward(self):
        output, labels = self.cache
        n = len(labels)
        output = softmax(output)
        output[np.arange(n), labels]-=1
        output /= n
        return output


