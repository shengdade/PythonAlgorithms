""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    w = weights[:-1]
    w0 = weights[-1]
    linear = np.dot(w.T, data.T) + w0
    y = sigmoid(linear.T)
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.x
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    ce = np.sum(-targets * np.log(y) - (1 - targets) * np.log(1 - y))
    frac_correct = float(np.sum((y >= 0.5).astype(np.int) == targets)) / targets.size
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    M = data.shape[1]
    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # compute f and df without regularization
        f = -np.sum(targets * np.log(y)) - np.sum((1 - targets) * np.log(1 - y))
        df = np.zeros((M + 1, 1))
        for j in xrange(M):
            df[j, 0] = np.sum(data[:, [j]] * (y - targets))
        df[M, 0] = np.sum(y - targets)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    M = data.shape[1]
    y = logistic_predict(weights, data)
    weight_decay = hyperparameters['weight_decay']

    # compute f and df with regularization
    negative_log_pw = 0.5 * weight_decay * np.sum(weights[:-1] ** 2) + M / 2 * np.log((2 * np.pi) / weight_decay)
    f = -np.sum(targets * np.log(y)) - np.sum((1 - targets) * np.log(1 - y)) + negative_log_pw
    df = np.zeros((M + 1, 1))
    for j in xrange(M):
        df[j, 0] = np.sum(data[:, [j]] * (y - targets)) + weight_decay * weights[j]
    df[M, 0] = np.sum(y - targets)

    return f, df
