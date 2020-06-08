from model import *
import sys
sys.path.append('neural_nets/')
from neural_nets import *
from loss_functions import SoftmaxLoss
from utilities import *
from training import GDOptimizer, Trainer

class LogisticRegression(Model):

    def __init__(self, input_dim, num_classes):
        self.classifier = neural_nets.Net()
        self.classifier.add(neural_nets.Linear(input_dim, num_classes))
        self.classifier.loss=SoftmaxLoss()

    def train(self, inputs, target, max_iter=10, **kwargs):
        self.config = kwargs
        optimizer = GDOptimizer(self.classifier, self.classifier.loss, update_rule='momentum', **self.config)
        dataset = Data(inputs, target)
        trainer = Trainer(dataset, optimizer, **self.config)
        trainer.train()

    def predict(self, inputs):
        logits = self.classifier(inputs)
        return np.argmax(logits, axis=1)

