import numpy as np

class LinearRegression(object):
    
    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.w)
        return y_pred

class Ridge(LinearRegression):
    
    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y

        l2 = 100
        self.w = np.linalg.solve(l2 * np.eye(X.T.shape[0]) + np.dot(X.T, X),
                                 np.dot(X.T, y))
    
class Lasso(LinearRegression):
    
    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y

        w = np.random.rand(X.shape[1]) / np.sqrt(X.shape[1])
        learning_rate = 0.001
        l1 = 10

        for i in range(100):
            y_pred = np.dot(X, w)
            delta = y_pred - y
            w -= (learning_rate * (l1 * np.sign(w) + np.dot(X.T, delta)))
        
        self.w = w
    
class ElasticNet(LinearRegression):
    
    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y

        w = np.random.rand(X.shape[1]) / np.sqrt(X.shape[1])
        learning_rate = 0.001
        l1 = 10

        for i in range(100):
            y_pred = np.dot(X, w)
            delta = y_pred - y
            w -= (learning_rate * (l1 * np.sign(w) + np.dot(X.T, delta)))
        
        l2 = 100

        self.w = np.linalg.solve(l2 * np.eye(X.T.shape[0]) +
                                 w + np.dot(X.T, X), np.dot(X.T, y))
