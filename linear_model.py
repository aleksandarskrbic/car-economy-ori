import numpy as np
import stats

class LinearRegression(object):
    
    def fit(self, X, y, X_test, y_test):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        
    def predict(self):
        y_pred = np.dot(self.X_test, self.w)
        return y_pred

class Ridge(LinearRegression):
    
    def fit(self, X, y, X_test, y_test):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test

        l2 = 100
        self.w = np.linalg.solve(l2 * np.eye(X.T.shape[0]) + np.dot(X.T, X),
                                 np.dot(X.T, y))
    
class Lasso(LinearRegression):
    
    def fit(self, X, y, X_test, y_test):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.w = calculate_optimal_weights(X, y, X_test, y_test)
    
class ElasticNet(LinearRegression):
    
    def fit(self, X, y, X_test, y_test):
        assert(X.shape[0] == y.shape[0])

        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        
        optimal_weights = calculate_optimal_weights(X, y, X_test, y_test)        
        l2 = 100

        self.w = np.linalg.solve(l2 * np.eye(X.T.shape[0]) +
                                 optimal_weights + np.dot(X.T, X), np.dot(X.T, y))

def calculate_optimal_weights(X, y, X_test, y_test):
    w = np.random.rand(X.shape[1]) / np.sqrt(X.shape[1])
    learning_rate = 0.001
    l1 = 10
    r_squared_list = []

    for i in range(500):
        y_pred = np.dot(X, w)
        delta = y_pred - y
        r2 = stats.get_r_squared(y_test, np.dot(X_test, w))
        r_squared_list.append((r2, w))
        w -= (learning_rate * (l1 * np.sign(w) + np.dot(X.T, delta)))
        
    optimal_weights = max(r_squared_list)[1]
    return optimal_weights
