from decision_tree import *
import scipy.io

path_train = 'datasets/spam-dataset/spam_data.mat'
data = scipy.io.loadmat(path_train)
X = data['training_data']
y = np.squeeze(data['training_labels'])
class_names = ["Ham", "Spam"]

tree = DecisionTree(impurity='entropy')
tree.train(X, y)
tree.visualize('test.pdf')
