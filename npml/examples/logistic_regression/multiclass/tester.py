import numpy as np
import sys
sys.path.append('../../../')
from logistic_regression import LogisticRegression
from model import Data
from utilities import *


train_data = np.load('data/train_data.npy')
test_data = np.load('data/test_data.npy')

train_inputs, train_target = Data.normalize(train_data[:,:-1]), train_data[:,-1:].astype(int).flatten()
test_inputs, test_target = Data.normalize(test_data[:,:-1]), test_data[:,-1:].astype(int).flatten()

model = LogisticRegression(input_dim=7, num_classes=3, batch_size=8, epochs=50, learning_rate=1e-3)
model.train(train_inputs, train_target)

print('After training the model accuracy is about ', accuracy(model.predict(test_inputs), test_target))
confusion_plot(model, test_inputs, test_target, outfile='plots/confusion_matrix')

