import pandas as pd
import numpy as np

dataset_path = 'auto-mpg.data'
column_names=['MPG', 'Cylinders', 'Displacement', 'Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset['Origin']=dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
train_dataset=dataset.sample(frac=0.8, random_state=0)
test_dataset=dataset.drop(train_dataset.index)
train_dataset = train_dataset.to_numpy()
test_dataset = test_dataset.to_numpy()
np.save('train_data', train_dataset)
np.save('test_data', test_dataset)
