from math import sqrt

def get_mean(x):
    return sum(x) / len(x)

def get_std(x):
    n = len(x)
    square_error_list = [(item - get_mean(x)) ** 2 for item in x]
    square_error_sum = sum(square_error_list)
    variance = square_error_sum / (n - 1)
    std = sqrt(variance)
    return std;

def get_pearson_corr(X, Y):
    assert(len(X) == len(Y))
    errors = [(x - get_mean(X)) * (y - get_mean(Y)) for x, y in zip(X, Y)]
    errors_sum = sum(errors)
    std_product = get_std(X) * get_std(Y)
    return (errors_sum / std_product) * (1 / (len(X) - 1))

def get_r_squared(y_test, y_predict):
    d1 = y_test - y_predict
    d2 = y_test - y_test.mean()
    return 1 - d1.dot(d1) / d2.dot(d2)

def standardize(X, y):
    for i in range(0, X.shape[1]):
         X[:, i] = (X[:, i] - get_mean(X[:, i].tolist())) / get_std(X[:, i].tolist())
    y = (y - get_mean(y.tolist())) / get_std(y.tolist())
    return X, y