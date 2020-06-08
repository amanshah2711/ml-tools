import numpy as np
from model import *

class KNN(Model):

    def train(self, inputs, labels):
        self.data = Data(inputs, labels)


    def predict(self, X):
        pass
