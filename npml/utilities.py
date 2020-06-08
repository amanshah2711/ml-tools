import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def softmax(X):
    X = X.T
    numer = np.exp(X - np.amax(X, axis=0))
    return (numer / np.sum(numer, axis=0)).T

def confusion_matrix(outputs, target):
    #Assume same labels are used in predictions and given labels
    num_classes = len(np.unique(target))
    cm = np.zeros((num_classes, num_classes))
    for i in range(len(target)):
        cm[outputs[i]][target[i]] += 1
    return cm

def confusion_plot(outputs, target, outfile=None):
    cm = confusion_matrix(outputs, target)
    plot = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion Matrix')
    if outfile:
        plot.figure.savefig(outfile, dpi=500)
    return plot

def sgd_update(param, dparam, config, prev):
    if not config['configured']:
        config.setdefault('learning_rate', 1e-3)
        config['configured']=True
    lr = config['learning_rate']
    param -= lr * dparam

def logits_to_classes(logits):
    return np.argmax(logits, axis=1)

def sgd_momentum_update(param, dparam, config, prev):
    if not config['configured']:
        config.setdefault('learning_rate', 1e-3)
        config.setdefault('momentum', 0.9)
        config['configured']=True
    lr, momentum, prev_update= config['learning_rate'], config['momentum'], prev[config['index']]
    update = momentum * prev_update - lr * dparam
    config['velocity']=update
    param += update
    return update

def rmsprop_update(param, dparam, config, prev):
    if not config['configured']:
        config.setdefault('learning_rate', 1e-3)
        config.setdefault('decay', 0.9)
        config.setdefault('epsilon', 1e-7)
        config['configured']=True
    lr, decay, eps, accum = config['learning_rate'], config['decay'], config['epsilon'], prev[config['index']]
    accum = decay * accum + (1 - decay) * (dparam**2)
    update = lr * dparam / np.sqrt(eps + accum)
    param -= update
    return accum

def adagrad_update(param, dparam, config, prev):
    if not config['configured']:
        config.setdefault('learning_rate', 1e-2)
        config.setdefault('epsilon', 1e-7)
        config['configured']=True
    lr, eps, accum = config['learning_rate'], config['epsilon'], prev[config['index']]
    accum += dparam ** 2
    update = lr * dparam / np.sqrt(accum + eps)
    param -= update
    return accum


def adam_update(param, dparam, config, prev):
    if not config['configured']:
        config.setdefault('learning_rate', 1e-3)
        config.setdefault('decay1', 0.9)
        config.setdefault('decay2', 0.999)
        config.setdefault('epsilon',1e-7)
        for i in range(config['index'] + 1):
            prev[i] = (1, 0, 0)
        config['configured'] = True
    lr, decay1, decay2, eps = config['learning_rate'], config['decay1'], config['decay2'], config['epsilon']
    time_step, first_moment, second_moment = prev[config['index']]
    first_moment = decay1 * first_moment + (1 - decay1) * dparam
    second_moment = decay2 * second_moment + (1 - decay2) * (dparam ** 2)
    first_unbias, second_unbias = first_moment / np.sqrt(1 - decay1 ** time_step), second_moment / np.sqrt(1 - decay2 ** time_step)
    update = lr * first_unbias / (np.sqrt(second_unbias) + eps)
    param -= update
    return (time_step + 1, first_moment, second_moment)

def accuracy(preds, labels):
    return (len(labels) - np.count_nonzero(preds -  labels)) / len(labels)

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
