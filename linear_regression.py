import numpy as np
from model import Model

identity_map = lambda x, y: x.T @ y
dual_auto = 0
cov_helper = lambda A: 0 if not A else np.linalg.inv(A)


class LinearRegression(Model):

    def __init__(self):
        self.weights = 0
        self.bias = 0

    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return X @ self.weights + self.bias  # should broadcast properly but double check with some testing


class LinearGaussianLS(LinearRegression):

    def __init__(self, prior_mean, prior_covariance, error_covariance, dual=dual_auto, kernel=identity_map):
        LinearRegression.__init__(self)
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.error_covariance = error_covariance  # error mean is usually assumed to be zero in general
        self.dual = dual
        self.kernel = kernel

    def train(self, X, y):
        n = X.shape[0]
        d = X.shape[1]
        X = np.hstack(X, np.ones((n, 1)))
        self.weights = self.prior_mean + np.linalg.inv(
            X.T @ self.error_covariance @ X + cov_helper(self.prior_covariance)) @ X.T @ np.linalg.inv(
            self.error_covariance) @ (y - X @ self.prior_mean)
        self.weights, self.bias = self.weights[:, :self.weights.shape[1] - 1], self.weights[:,
                                                                               self.weights.shape[1] - 1:]


class GLS(LinearGaussianLS):
    def __init__(self, error_covariance, dual=dual_auto, kernel=identity_map):
        LinearGaussianLS.__init__(self, prior_mean=0, prior_covariance=0, error_covariance=error_covariance)

    def train(self, X, y):
        # see if there is a clean reduction to OLS that allows OLS method use
        n = X.shape[0]
        d = X.shape[1]
        X = np.hstack(X, np.ones((n, 1)))
        sig_inv = np.linalg.inv(self.error_covariance)



class WLS(GLS):
    def __init__(self, error_covariance, dual=dual_auto):
        GLS.__init__(self, error_covariance=np.diag(error_covariance))


class OLS(WLS):

    def __init__(self, dual=dual_auto, kernel=identity_map):
        WLS.__init__(self, error_covariance=1)
        self.dual = dual
        self.kernel = kernel

    def train(self, X, y):
        n = X.shape[0]
        #d = X.shape[1]
        #X = np.hstack((X, np.ones((n, 1))))
        self.error_covariance = np.identity(n)
        super().train(X, y)
        """
        if self.dual == dual_auto and n >= d:
            self.dual = False
        else:
            self.dual = True

        if not self.dual:
            self.weights = np.matmul(np.linalg.inv(X.T @ X) @ X.T, y)
        else:
            self.weights = X.T @ np.linalg.inv(X @ X.T) @ y
        """


class TikhonovRegularization(LinearGaussianLS):
    def __init__(self, prior_mean, prior_covariance):
        LinearGaussianLS.__init__(self, prior_mean=prior_mean, prior_covariance=prior_covariance, error_covariance=1)

    def train(self, X, y):
        n = X.shape[0]
        self.error_covariance = np.identity(n)
        super().train(X, y)

class RidgeRegression(TikhonovRegularization):

    def __init__(self, alpha, dual=dual_auto, kernel=identity_map):
        TikhonovRegularization.__init__(self, prior_mean=0, prior_covariance=alpha)
        self.dual = dual
        self.kernel = kernel
        self.alpha = alpha

    def train(self, X, y):
        d = X.shape[1]
        self.prior_covariance = self.alpha * np.identity(d)
        super().train(X,y)
        """
        d = X.shape[1]
        X = np.hstack((X, np.ones((n, 1))))
        if self.dual == dual_auto and n >= d:
            self.dual = False
        else:
            self.dual = True

        if not self.dual:
            self.weights = np.matmul(np.linalg.inv(X.T @ X + self.alpha * np.identity(d)) @ X.T, y)
        else:
            self.weights = X.T @ np.linalg.inv(X @ X.T) @ y  # replace with kernel ridge regression solution
            """

class TotalLS(LinearRegression):

    def __init__(self):
