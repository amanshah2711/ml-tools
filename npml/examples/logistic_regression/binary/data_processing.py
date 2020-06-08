import pandas as pd
import numpy as np

data_path = 'data/pima-indians-diabetes.csv'
df = pd.read_csv(data_path)

print(df)
data = df.to_numpy()

np.random.shuffle(data)

split = len(data) * 3 // 4
train_data, test_data = data[:split,:], data[split:,:]
print('The number of training points is %d' % (len(train_data)))
print('The distribution of training points is ', np.bincount(train_data[:,-1:].astype(int).flatten()))
print()

print('The number of testing points is %d' % (len(test_data)))
print('The distribution of testing points is ', np.bincount(test_data[:,-1:].astype(int).flatten()))
print()

print('The shape of the an input point is, ', train_data[0].shape)

np.save('data/train_data', train_data)
np.save('data/test_data', test_data)
