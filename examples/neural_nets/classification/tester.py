import sys
sys.path.append('../../../neural_nets')
sys.path.append('../../..')
from neural_nets import *
from loss_functions import *
from training import *
import seaborn as sns
from utilities import *
import matplotlib.pyplot as plt
sns.set()

#Loads the data
train_data, test_data = np.load('data/train_data.npy'), np.load('data/test_data.npy')
train_inputs, train_labels = train_data[:,:-1], train_data[:,-1:]
test_inputs, test_labels= test_data[:,:-1], test_data[:,-1:]

#Friendly Wrapper for Data
train_data = Data(train_inputs, train_labels.flatten())
test_data = Data(test_inputs, test_labels.flatten())

#Setting up the model
model = Net(Linear(784, 128), ReLU(), BatchNorm(128), Linear(128, 10))
#model.add()

#Setting up training
optimizer = GDOptimizer(model, SoftmaxLoss(), update_rule='sgd')
trainer = Trainer(train_data, optimizer, batch_size = 32, epochs=5)
train_loss = trainer.train()

plt.plot(list(range(6)), train_loss, label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('plots/loss_curve', dpi=400)

#Fun stuff
test_outputs = model(test_data.inputs)
test_outputs = np.argmax(test_outputs, axis=1)
confusion_plot(test_outputs, test_data.target, 'plots/confusion_matrix')

print('Test data shows we are accurate, %.2f of the time + \n' % (accuracy(test_data.target, test_outputs)))

"""
for epoch in range(epochs):
    for iteration in range(num_iterations):
        batch_inputs, batch_labels = train_data.build_batch(batch_size)
        output = model(batch_inputs)
        loss = cross_entropy(output, batch_labels)
        optimizer.step()
        if iteration == 0:
            print('[Epoch %d / %d ] Iteration %d / %d Training loss: %.2f ' % (epoch+1, epochs, iteration, num_iterations-1, loss))

correct = np.count_nonzero(test_outputs - test_data.outputs)

test_outputs = model(train_data.inputs)
test_outputs = np.amax(test_outputs, axis=1)
correct = np.count_nonzero(test_outputs - train_data.outputs)
print('Training data shows we are accurate, %.2f of the time + \n' % (correct/ len(train_data)))
"""

