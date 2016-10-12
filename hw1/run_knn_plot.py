import matplotlib.pyplot as plt
from run_knn import run_knn
from utils import *


def run_knn_plot():
    # load train data and valid data
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()

    # count classification rate for each k
    kk = range(1, 10, 2)
    classification_rate = [count_rate(k, train_inputs, train_targets, valid_inputs, valid_targets) for k in kk]
    print("classification_rate", classification_rate)

    # plot the rates
    plt.plot(kk, classification_rate, 'ro', kk, classification_rate, 'y')
    plt.axis([0, 10, 0.8, 1])
    plt.xlabel('k')
    plt.ylabel('classification rate')
    plt.show()


def count_rate(k, train_inputs, train_targets, valid_inputs, valid_targets):
    valid_labels = run_knn(k, train_inputs, train_targets, valid_inputs)
    return float(np.sum(valid_labels == valid_targets)) / valid_targets.size


if __name__ == "__main__":
    run_knn_plot()
