from __future__ import print_function
from util import Load, LoadData
from nn import Softmax, NNForward
from cnn import CNNForward
import numpy as np
import matplotlib.pyplot as plt

plt.ion()


def count_probability(x, model_fname, forward):
    model = Load(model_fname)
    var = forward(model, x)
    return Softmax(var['y'])


def nn_predict(x):
    args = ([16, 32],  # num_hiddens
            0.01,  # eps
            0.0,  # momentum
            1000,  # num_epochs
            100)  # batch_size
    fname = str(args[0]) + '_' + str(args[1]) + '_' + str(args[2]) + '_' + str(args[3]) + '_' + str(args[4])
    return count_probability(x, 'nn_model/' + fname + '.npz', NNForward)


def cnn_predict(x):
    args = (0.1,  # eps
            0.0,  # momentum
            30,  # num_epochs
            5,  # filter_size
            8,  # num_filters_1
            16,  # num_filters_2
            100)  # batch_size
    fname = str(args[0]) + '_' + str(args[1]) + '_' + str(args[2]) + '_' + str(args[3]) + '_' + \
            str(args[4]) + '_' + str(args[5]) + '_' + str(args[6])
    return count_probability(x, 'cnn_model/' + fname + '.npz', CNNForward)


def plot_faces(x, fname):
    plt.clf()
    for i in xrange(x.shape[0]):
        plt.subplot(1, x.shape[0], i + 1)
        plt.imshow(x[i, :].reshape(48, 48), cmap=plt.cm.gray)
    plt.draw()
    plt.savefig('uncertainty/' + fname)
    print('uncertain expressions saved to: uncertainty/' + fname + '.png')


def find_uncertain(data, start, end, method='NN'):
    if method == 'NN':
        predict = nn_predict
    else:
        predict = cnn_predict
    threshold = 0.5
    x = data[start:end + 1]
    if x.ndim == 1:
        x = x.reshape(1, -1)
    max_prob = np.max(predict(x), axis=1)
    index_uncertain = start + np.nonzero((max_prob < threshold).astype(int))[0]
    print('prediction prob: ' + str(max_prob))
    print('uncertain index: ' + str(index_uncertain))
    plot_faces(x, str(index_uncertain))


def main():
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('../toronto_face.npz')
    find_uncertain(inputs_test, 33, 33, 'NN')


if __name__ == '__main__':
    main()
