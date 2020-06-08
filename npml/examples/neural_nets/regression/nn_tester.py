import npml
import numpy as np
from npml.neural_net.neural_nets import *
from npml.utils.utilities import softmax
from npml.model import Data
from npml.neural_net.loss_functions import MSELoss
print('done')

#Defining a basic feedforward network
class FeedForward(Net):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = [Linear(input_dim=input_dim, output_dim=64), ReLU(), Linear(input_dim=64, output_dim=64), ReLU(), Linear(input_dim=64, output_dim=1)]


#Loading a dataset
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

train_data, train_labels = train_data[:,1:], train_data[:,0:1]
test_data, test_labels = test_data[:,1:], test_data[:,0:1]

#Wrapping the data with some friendly tools
train_data = Data(train_data, train_labels, normalize=True)
test_data = Data(test_data, test_labels, normalize=True)
print('Number of training samples', len(train_data))
print('Number of test samples', len(test_data))

#Setting up the model
model = Net()
model.add(Linear(input_dim=9, output_dim=64))
model.add(ReLU())
model.add(Linear(input_dim=64, output_dim=64))
model.add(ReLU())
model.add(Linear(input_dim=64, output_dim=1))

#Setting up the optimizer 
lr, mse, epochs = 1e-3, MSELoss(), 100
optimizer = GDOptimizer(model, cost_function=mse, update_rule='momentum', momentum=0.85)

#Training Loop
batch_size = 32
num_iterations = len(train_data) // batch_size #NOTE:Define a testing dataset
train_loss, loss = [], 0
for epoch in range(epochs):
    for iteration in range(num_iterations):
        batch_input, batch_labels = train_data.build_batch(batch_size)
        output = model(batch_input)
        loss = mse(output, batch_labels)
        if  iteration == 0:
            print('[Epoch %d / %d ] Iteration %d / %d Training loss: %.2f ' % (epoch+1, epochs, iteration, num_iterations-1, loss))
        optimizer.step()
    train_loss.append(loss)


plot = sns.scatterplot(x=[i for i in range(epochs)], y = train_loss)
plot.set(ylim=(0, 100))
plot.figure.savefig('output.png')

