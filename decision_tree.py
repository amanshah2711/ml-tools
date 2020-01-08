from model import Model
import numpy as np
from utilities import entropy, gini_impurity, mutual_information, gini_purification


class DecisionTree(Model):
    impurity_funcs = {'gini': gini_impurity,
                      'entropy': entropy}
    purification = {'gini': gini_purification,
                    'entropy': mutual_information}

    # expand color labels to be larger so it generalizes
    color_labels = {0: 'turquoise',
                    1: 'orange',
                    2: 'red'}

    def __init__(self, impurity='gini', max_depth=-1):
        self.max_depth = max_depth
        self.impurity_metric = impurity
        self.thresh = self.index = self.purity = self.category = -1
        self.left = self.right = None
        self.id = ''

    def is_leaf(self):
        """
        Determines if a node in this tree is a leaf(i.e that is has no branches)
        :return: Boolean value
        """
        return not self.left and not self.right

    def segmenter(self, X, y):
        """
        Finds the best possible split in the data that maximizes gain according to
        the designated impurity function
        :param X: input n x d  data, or design matrix
        :param y: n x 1 vector of labels
        :return: 1 x 3 list of values with index, and threshold of optimal split with gain attained
        """
        max_gain = gain = -1
        index = opt_thresh = -1
        purity = DecisionTree.impurity_funcs[self.impurity_metric]
        gain_metric = DecisionTree.purification[self.impurity_metric]
        for col in np.arange(X.shape[1]):
            col_x = X[:, col: col + 1].flatten()
            for thresh in np.unique(col_x):
                gain = gain_metric(col_x, y, thresh)
                if gain > max_gain:
                    max_gain = gain
                    index = col
                    opt_thresh = thresh
        return [index, opt_thresh, max_gain]

    def split(self, X, y, index, thresh):
        """
        Splits the data according to the index values w.r.t threshold
        :param X: the n x d data, or design, matrix
        :param y: the n x 1 label vector
        :param index: integer denoting component that splits
        :param thresh: float that divides the data
        :return: 1 x 4 list with the split labels and data
        """
        # getting the column and properly reshaping
        col_x = X[:, index: index + 1].flatten()

        # Finding data points that adhere to split
        indicator_below = np.where(col_x < thresh)[0]
        indicator_above = np.where(col_x >= thresh)[0]

        # Constructing the partitioned data and labels
        data_below = np.take(X, indicator_below, axis=0)
        data_above = np.take(X, indicator_above, axis=0)
        label_below = np.take(y, indicator_below)
        label_above = np.take(y, indicator_above)

        return [data_below, label_below, data_above, label_above]

    def train(self, X, y):
        """
        Takes in training data and learns the optimal tree w.r.t the training data
        :param X: n x d data, or design, matrix
        :param y: n x 1 vector of labels
        :return: float that is training error achieved on optimal tree
        """
        max_d = 0
        error = 1
        for d in np.arange(15) + 1:
            training_error = self.train_internal(X, y, d)
            print(d)
            if error > training_error:
                max_d = d

        print('\n')
        print(max_d)
        return self.train_internal(X, y, max_d)

    def train_internal(self, X, y, depth=0):
        self.index, self.thresh, gain = self.segmenter(X, y)
        self.purity = DecisionTree.impurity_funcs[self.impurity_metric]
        self.category = np.argmax(np.bincount(y))
        data_below, label_below, data_above, label_above = self.split(X, y, self.index, self.thresh)

        if data_above.size and data_below.size and depth > 0 and gain > 0:
            self.left = DecisionTree()
            self.left.id = self.id + 'l'
            self.right = DecisionTree()
            self.right.id = self.id + 'r'
            self.left.train_internal(data_below, label_below, depth - 1)
            self.right.train_internal(data_above, label_above, depth - 1)

        train_error = self.error(X, y)
        return train_error

    def predict_single(self, input):
        """
        predicts category for input where input is presumed to be one data point
        :param input: a 1 x d vector that is input to be classified
        :return: int corresponding to a category
        """
        input = input.flatten()
        if not self.left and not self.right:
            return self.category
        elif input[self.index] < self.thresh:
            return self.left.predict_single(input)
        else:
            return self.right.predict_single(input)

    def predict(self, X):
        """
        Predicts category for multiple input data points stored in a data matrix.
        :param X: n x d data, or design, matrix
        :return:  an n element numpy array of values
        """
        y_hat = np.zeros(X.shape[0])
        for row in np.arange(X.shape[0]):
            row_x = X[row: row + 1, :].flatten()
            y_hat[row] = self.predict_single(row_x)

        return y_hat

    def __str__(self):
        return 'index: ' + str(self.index) + '\n' + 'threshold: ' + str(
            np.round(self.thresh, 2)) + '\n' + 'category: ' + str(self.category)

    def prep_visual(self):
        """
        Defines the nodes and edges in the graph representing the tree structure.
        :return: None
        """
        if self.is_leaf():
            node = 'category: ' + str(self.category)
            self.graph.node(self.id, node, fillcolor=DecisionTree.color_labels[self.category], style='filled')
        else:
            node = str(self)
            self.graph.node(self.id, node)
            self.left.graph = self.graph
            self.left.prep_visual()
            self.right.graph = self.graph
            self.right.prep_visual()
            self.graph.edge(self.id, self.left.id)
            self.graph.edge(self.id, self.right.id)

    def graph_init(self, filetype):
        """
        Helper method to visualize designed to create the necessary infrastructure for prep_visual()
        :param filetype: string corresponding to desired output filetype
        :return: None
        """
        from graphviz import Digraph
        self.graph = Digraph(format=filetype)
        self.graph.attr('node', shape='ellipse')

    def visualize(self, outfile='', dir=''):
        """
        This is a method to be called by the user that actually renders and outputs the graph
        of the tree to the file, outfile.
        :param outfile: string of filename desired for output file
        :return: None
        """
        outfile = outfile.split('.')
        self.graph_init(outfile[1])
        self.prep_visual()
        self.graph.render(filename=outfile[0], directory=dir)

    class RandomForests(Model):

        def __init__(self, impurity_metric, max_depth, num_trees=100, bagging=True, bootstrap_sample_size=-1,
                     feature_sample_size=0.6):  # TODO make this a parameter a good default
            self.trees = [DecisionTree(impurity_metric, max_depth) for _ in np.arange(num_trees)]
            self.bagging = bagging
            self.bootstrap_sample_size = bootstrap_sample_size
            self.feature_sample_size = np.min(np.max(0, feature_sample_size), 1)

        def train(self, X, y):
            if self.bagging:
                if self.bootstrap_sample_size == -1:
                    self.bootstrap_sample_size = X.shape[0]
                for index in np.arange(len(self.trees)):
                    data_chosen = np.random.randint(0, high=X.shape[0], size=self.bootstrap_sample_size)
                    X_prime = np.take(X, data_chosen, axis=0)
                    y_prime = np.take(y, data_chosen)
                    self.trees[index].train(X_prime, y_prime)
            else:
                # implement feature bagging
                assert self.feature_sample_size > 0
                d = X.shape[1]
                if self.feature_sample_size == 1:
                    print(
                        "WARNING: Utilizing all features for every decision tree eliminates benefits of random forest "
                        "use")

                for index in np.arange(len(self.trees)):
                    features = np.random.randint(d, np.floor(self.feature_sample_size * d))
                    X_prime = np.take(X.T, features, axis=0).T
                    self.trees[index].train(X_prime, y)

        def predict_single(self, X):
            return np.argmax(np.bincount(np.array([dtree.predict_single(X) for dtree in self.trees])))

        def predict(self, input):
            predictions = np.array([self.predict_single(row) for row in input])
            return predictions

        def visualize(self, outfile=''):
            import os
            cwd = os.getcwd()
            dir = cwd + outfile

            if not os.path.exists(dir):
                os.mkdir(dir)

            for i in np.arange(len(self.trees)):
                name = 'tree' + str(i) + '.png'  # make some structural tweaks to allow more general file choice
                self.trees[i].visualize(name, dir)
