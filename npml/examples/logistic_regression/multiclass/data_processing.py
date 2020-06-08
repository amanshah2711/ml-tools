import pandas as pd
import numpy as np

df = pd.read_csv('data/seeds.csv')
print(df)

data = df.to_numpy()
np.random.shuffle(data)
data[:,-1:] -= 1

split = len(data) * 3 // 4
train_data, test_data = data[:split, :], data[split:, :]
print('The number of training points is ', len(train_data))
print('The number of testing points is ', len(test_data))
print('A point has shape like, ', train_data[0].shape)
print('The distribution of training points is, ', np.bincount(train_data[:,-1:].astype(int).flatten()))
print('The distribution of testing points is, ', np.bincount(test_data[:,-1:].astype(int).flatten()))

np.save('data/train_data', train_data)
np.save('data/test_data', test_data)
