class Layer:

    def __init__(self, activation_function):
        self.activation = activation_function

class Net(Model):

    def __init__(self):
        self.layers = []
