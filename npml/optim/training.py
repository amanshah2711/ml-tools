from ..model import Optimizer
from ..utils.utilities import *

class GDOptimizer(Optimizer):

    update_rules = {'sgd' : sgd_update,
                    'momentum': sgd_momentum_update,
                    'rmsprop': rmsprop_update,
                    'adagrad': adagrad_update,
                    'adam': adam_update
                    }

    def __init__(self, model, cost_function, update_rule = 'sgd', **kwargs):
        self.model = model
        self.params=[]
        for layer in reversed(model.layers):
            self.params.extend(layer.params)
        self.config=kwargs
        self.config['configured'] = False
        self.cost_function = cost_function
        self.prev_state = [0]*len(self.params)
        self.update_rule = GDOptimizer.update_rules[update_rule]

    def step(self):
        dloss = self.cost_function.backward()
        grads = self.model.backward(dloss)
        for param, grad, index in zip(self.params, grads, reversed(np.arange(len(self.prev_state)))):
            self.config['index'] = index
            new_state = self.update_rule(param, grad, self.config, self.prev_state)
            self.prev_state[index] = new_state

    def _clear_cache(self):
        pass

class Trainer(object):

    def __init__(self, train_data, optimizer=None, **kwargs):
        self.train_data = train_data
        self.optimizer = optimizer
        self.config = kwargs
        self.config.setdefault('epochs', 10)
        self.config.setdefault('batch_size', 16)
    def train(self):
        batch_size, epochs, loss_func = self.config['batch_size'], self.config['epochs'], self.optimizer.cost_function
        num_iterations = len(self.train_data) // batch_size
        train_loss = []
        for epoch in range(epochs):
            for iteration in range(num_iterations):
                batch_inputs, batch_target = self.train_data.build_batch(batch_size)
                batch_outputs = self.optimizer.model(batch_inputs)
                loss = loss_func(batch_outputs, batch_target)
                self.optimizer.step()
                if epoch == 0 and iteration == 0:
                    train_loss.append(loss)
                    _reporter(0, num_iterations, epoch + 1, epochs, loss)
            train_loss.append(loss)
            _reporter(iteration + 1, num_iterations, epoch + 1, epochs, loss)
        print('Finished Training')
        return train_loss

def _reporter(curr_iter, max_iter, curr_epoch, max_epoch, loss, loss_type='Training'):
    """TODO: Docstring for _reporter.
    :returns: TODO

    """
    print('[ %d / %d ] epochs [ %d / %d ] iterations | %s loss is %.2f' % (curr_epoch, max_epoch, curr_iter, max_iter, loss_type, loss))

