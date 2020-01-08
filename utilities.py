import numpy as np


def entropy(y):
    """ Calculates entropy, in the information theoretic sense, of the input vector.

    :param y: input n x 1 vector
    :return: entropy of y
    """
    total = y.size
    counts = np.bincount(y)
    counts = counts[counts != 0] / y.size
    result = np.sum(-counts * np.log2(counts))
    return result


def gini_impurity(y):
    """
    Calculates Gini impurity of the labels
    :param y: n x 1 vector
    :return: Gini impurity
    """
    total = y.size
    counts = np.bincount(y)
    counts = np.square(counts[counts != 0] / total)
    return 1 - np.sum(counts)


def mutual_information(X, y, thresh):
    """
    Calculates mutual information(also called information gain), in the information theoretic sense,
     where X is the column vector to be conditioned on with threshold splitting X.
    :param X: n x 1 vector
    :param y: n x 1 vector of labels
    :param thresh: float number to be split on
    :return: float mutual information
    """
    X = X.flatten()
    total = len(X)
    ent = entropy(y)

    # Creating useful indicator indices
    indicator_below = np.array(np.where(X < thresh)[0])
    indicator_above = np.array(np.where(X >= thresh)[0])

    # Finding Number below and above the threshold
    num_below = indicator_below.size
    num_above = indicator_above.size

    # Calculating split probabilities
    prob_above = num_above / total
    prob_below = num_below / total

    # Calculate conditional entropies
    below = np.take(y, indicator_below)
    above = np.take(y, indicator_above)
    entr_above = entropy(above)
    entr_below = entropy(below)

    # Calculating Gain
    gain = ent - prob_below * entr_below - prob_above * entr_above
    return gain


def gini_purification(X, y, thresh):
    """
    Calculating the gain w.r.t the gini impurity for a given split at thresh of points in X.
    :param X: n x 1 vector
    :param y: n x 1 vector
    :param thresh: float
    :return: float
    """

    X = X.flatten()
    total = len(y)
    gini = gini_impurity(y)

    # Creating useful indicator indices
    indicator_below = np.array(np.where(X < thresh)[0])
    indicator_above = np.array(np.where(X >= thresh)[0])

    # Finding Number below and above the threshold
    prob_below = indicator_below.size / total
    prob_above = indicator_above.size / total

    # Calculate conditional impurities
    below = np.take(y, indicator_below)
    above = np.take(y, indicator_above)
    imp_above = gini_impurity(above)
    imp_below = gini_impurity(below)

    # reduction
    reduction = gini - prob_above * imp_above - prob_below * imp_below

    return reduction
