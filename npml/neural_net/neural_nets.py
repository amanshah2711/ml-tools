import numpy as np
import sys
sys.path.append('../')
from model import *

TRAIN = 'train'
TEST = 'test'


class Layer(object):

    def __init__(self):
        self.params=[]
        self.mode = TRAIN

    def __call__(self, inputs):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

class Linear(Layer):

    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        k = np.sqrt(1/input_dim)
        self.W = np.random.uniform(-k, k, size=(output_dim, input_dim))
        self.b = np.random.uniform(-k, k, size=(output_dim))
        self.params = [self.W, self.b]

    def __call__(self, inputs):
        self.cache = inputs
        out = inputs @ self.W.T
        if self.use_bias:
            out += self.b
        return out

    def backward(self, dout):
        inputs = self.cache
        db, dW, dinputs = np.sum(dout, axis=0), (inputs.T @ dout).T, dout @ self.W
        return dW, db, dinputs

class ReLU(Layer):

    def __call__(self, inputs):
        self.cache = inputs
        return np.maximum(0, inputs)

    def backward(self, dout):
        inputs = self.cache
        dinputs = dout * (inputs > 0)
        return (dinputs,)

class Sigmoid(Layer):

    def __call__(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def backward(self, dout):
        dlayer = np.exp(-inputs) / (1+ np.exp(-inputs))**2
        return (dout * dlayer,)

class Softmax(Layer):

    def __call__(self, inputs):
        inputs = inputs.T
        numer = np.exp(inputs - np.amax(inputs, axis=0))
        return (numer / np.sum(numer, axis=0)).T

    def backward(self, dout):
        pass #NOTE: This is a tensor which I need to go and compute by hand first

class Tanh(Layer):

    def __call__(self, inputs):
        self.cache = np.tanh(inputs)
        return self.cache

    def backward(self, dout):
        tanh_inputs = self.cache
        dinputs = dout * (1 - tanh_inputs ** 2)
        return (dinputs,)

class Dropout(Layer):

    def __init__(self, dropout_keep_prob):
        super().__init__()
        self.dropout_keep_prob = dropout_keep_prob

    def __call__(self, inputs):
        if self.mode == TRAIN:
            mask = np.random.binomial(1, self.dropout_keep_prob, size=inputs.shape)
            self.cache = (inputs, mask)
            out = inputs * mask / self.dropout_keep_prob
        elif self.mode == TEST:
            out = inputs
        return out

    def backward(self, dout):
        if self.mode == TRAIN:
            inputs, mask = self.cache
            dinputs = dout * mask / self.dropout_keep_prob
        elif self.mode == TEST:
            dinputs = dout
        return (dinputs,)

class Conv(Layer):

    def __call__(self, inputs):
        pass

    def backward(self, dout):
        pass

class BatchNorm(Layer):

    def __init__(self, input_dim, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = 0.9
        self.eps = eps
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.zeros(input_dim)
        self.scale = np.ones(input_dim)
        self.shift = np.zeros(input_dim)
        self.params = [self.scale, self.shift]

    def __call__(self, inputs):
        if self.mode == TRAIN:
            mean = np.mean(inputs, axis=0)
            var = np.mean((inputs - mean) ** 2, axis=0)
            out = (inputs - mean) / np.sqrt(var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            self.cache = (inputs, out, mean, var)
        elif self.mode == TEST:
            out = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.cache = (inputs, out)
        out = self.scale * out + self.shift
        return out

    def backward(self, dout):
        if self.mode == TRAIN:
            inputs, prescale, mean, var = self.cache
            dinputs = dout * self.scale
            dinputs = dinputs - np.mean(dinputs, axis=0) - np.mean(dinputs * prescale, axis=0) * prescale
            dinputs /= np.sqrt(var + self.eps)
        elif self.mode == TEST:
            inputs, prescale = self.cache
            dinputs =  self.scale * dout / np.sqrt(self.running_var + self.eps)
        dscale = np.sum(dout * prescale, axis=0)
        dshift = np.sum(dout, axis=0)
        return dscale, dshift, dinputs

class SpatialBatchNorm(Layer):

    def __call__(self, inputs):
        pass

    def backward(self, dout):
        pass

class Net(Model):

    def __init__(self, *args):
        self.layers = []
        self.layers.extend(args)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def predict(self, inputs):
        return self.forward(inputs) #see if we should alias this with forward

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            cache = layer.backward(dout)
            dout = cache[-1]
            grads.extend(cache[:-1])
        return grads

    def add(self, *args):
        #assert isinstance(layer, Layer), 'input layer is not a valid layer'
        self.layers.extend(args)

    def train(self, inputs, target):
        pass

    def eval(self):
        for layer in self.layers:
            layer.mode=TEST
        #[(lambda x : x.mode = TEST)(layer) for layer in self.layers] #potential alternate implementation
