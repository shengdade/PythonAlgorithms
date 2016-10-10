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
    y = 1 - sigmoid(linear.T)
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
    ce = np.sum(targets * (-np.log(y)))
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

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # compute f and df without regularization
        f = -np.sum(targets * np.log(y)) - np.sum((1 - targets) * np.log(1 - y))
        df = np.zeros((weights.size, 1))
        for j in xrange(weights.size - 1):
            df[j, 0] = np.sum(data[:, [j]] * (targets - y))
        df[weights.size - 1, 0] = np.sum(targets - y)

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

    y = logistic_predict(weights, data)
    weight_decay = hyperparameters['weight_decay']

    # compute f and df with regularization
    f = -np.sum(targets * np.log(y)) - np.sum((1 - targets) * np.log(1 - y))
    df = np.zeros((weights.size, 1))
    for j in xrange(weights.size - 1):
        df[j, 0] = np.sum(data[:, [j]] * (targets - y)) + weight_decay * df[j, 0]
    df[weights.size - 1, 0] = np.sum(targets - y) + weight_decay * df[weights.size - 1, 0]

    return f, df
