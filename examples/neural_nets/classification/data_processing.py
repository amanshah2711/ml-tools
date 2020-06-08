from mlxtend.data import loadlocal_mnist
import numpy as np

train_inputs, train_labels = loadlocal_mnist(images_path = 'data/train-images-idx3-ubyte', labels_path = 'data/train-labels-idx1-ubyte')
test_inputs, test_labels = loadlocal_mnist(images_path = 'data/t10k-images-idx3-ubyte', labels_path = 'data/t10k-labels-idx1-ubyte')
num_train = train_inputs.shape[0]
num_test = test_inputs.shape[0]

print('Number of training points', num_train)
print('Number of test points', num_test)
print('Distribution of training labels', np.bincount(train_labels))

train_data = np.hstack((train_inputs, train_labels.reshape((num_train, 1))))
test_data = np.hstack((test_inputs, test_labels.reshape((num_test,1))))

np.save('data/train_data', train_data)
np.save('data/test_data', test_data)



