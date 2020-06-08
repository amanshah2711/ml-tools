import sys
sys.path.append('../')
import seaborn as sns
sns.set()
import torch
import torch.nn.functional as F
import numpy as np

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    return data

def build_batch(dataset, batch_size):
    inputs, labels = dataset
    indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
    return normalize(np.take(inputs, indices, axis=0)), np.take(labels, indices, axis=0)

#Defining a basic feedforward network
class FeedForward(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Loading a dataset
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

train_data, train_labels = train_data[:,1:], train_data[:,0:1]
test_data, test_labels = test_data[:,1:], test_data[:,0:1]

train_data = [normalize(train_data), train_labels]
test_data = [normalize(test_data), test_labels]
num_train = train_data[0].shape[0]
num_test = test_data[0].shape[0]
print('Number of training samples', num_train)
print('Number of test samples', num_test)

print('Data point shape', train_data[0][0].shape)

#Setting up the model
model = FeedForward(9, 1)

#Setting up the optimizer 
mse, epochs = torch.nn.MSELoss(), 100
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)

#Training Loop
batch_size = 16
num_iterations = num_train // batch_size #NOTE:Define a testing dataset
train_loss, loss = [], 0
for epoch in range(epochs):
    train_loss.append(loss)
    for iteration in range(num_iterations):
        batch_input, batch_labels = build_batch(train_data, batch_size)
        output = model(torch.Tensor(batch_input))
        optimizer.zero_grad()
        loss = mse(output, torch.Tensor(batch_labels))
        loss.backward()
        optimizer.step()
        if  iteration == 0:
            print('[Epoch %d / %d ] Iteration %d / %d Training loss: %.2f ' % (epoch+1, epochs, iteration+1, num_iterations, loss))


plot = sns.scatterplot(x=[i for i in range(epochs)], y = train_loss)
plot.set(ylim=(0, 100))
plot.figure.savefig('torch_output.png')

