import numpy as np
import sys
sys.path.append('../../../')
from logistic_regression import LogisticRegression
from model import Data
from utilities import *

train_data, test_data = np.load('data/train_data.npy'), np.load('data/test_data.npy')

train_inputs, train_target = train_data[:,:-1], train_data[:,-1].astype(int).flatten()
test_inputs, test_target= test_data[:,:-1], test_data[:,-1].astype(int).flatten()

model = LogisticRegression(input_dim=8, num_classes=2)
model.train(train_inputs, train_target, learning_rate=1e-4, batch_size=64, epochs=50)
test_outputs = model.predict(test_inputs)
confusion_plot(test_outputs, test_target, outfile='plots/confusion_matrix')
print('The accuracy of this trained model is %.2f' % (accuracy(test_outputs, test_target)))
