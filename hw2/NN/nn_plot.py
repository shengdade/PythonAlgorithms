from util import Load, Save
from nn import InitNN, CheckGrad, NNForward, NNBackward, NNUpdate, Train
import numpy as np
import matplotlib.pyplot as plt

# Turn interactive plotting off
plt.ioff()


def nn_with_args(num_hiddens=[16, 32],
                 eps=0.01,
                 momentum=0.0,
                 num_epochs=1000,
                 batch_size=100):
    # Input-output dimensions.
    num_inputs = 2304
    num_outputs = 7

    # Initialize model.
    model = InitNN(num_inputs, num_hiddens, num_outputs)

    # Check gradient implementation.
    print('Checking gradients...')
    x = np.random.rand(10, 48 * 48) * 0.1
    CheckGrad(model, NNForward, NNBackward, 'W3', x)
    CheckGrad(model, NNForward, NNBackward, 'b3', x)
    CheckGrad(model, NNForward, NNBackward, 'W2', x)
    CheckGrad(model, NNForward, NNBackward, 'b2', x)
    CheckGrad(model, NNForward, NNBackward, 'W1', x)
    CheckGrad(model, NNForward, NNBackward, 'b1', x)

    # Train model.
    model, stats = Train(model, NNForward, NNBackward, NNUpdate, eps, momentum, num_epochs, batch_size)

    # Construct file name
    fname = str(num_hiddens) + '_' + str(eps) + '_' + str(momentum) + '_' + str(num_epochs) + '_' + str(batch_size)

    # Save the training statistics.
    Save('nn_model/' + fname, model)
    Save('nn_stats/' + fname, stats)

    return fname


def plot_figures(stats_fname):
    stats = Load(stats_fname)
    train_ce_list = stats['train_ce']
    valid_ce_list = stats['valid_ce']
    train_acc_list = stats['train_acc']
    valid_acc_list = stats['valid_acc']
    figure_path = get_nn_path(stats_fname)
    save_figure(train_ce_list, valid_ce_list, 'Cross Entropy', figure_path)
    save_figure(train_acc_list, valid_acc_list, 'Accuracy', figure_path)
    print('Figures saved to ' + figure_path)


def get_nn_path(stats_fname):
    sep = stats_fname.find('/')
    figure_name = stats_fname[sep + 1:-4]
    return "nn_figure/" + figure_name


def save_figure(train, valid, ylabel, fname):
    train = np.array(train)
    valid = np.array(valid)
    plt.plot(train[:, 0], train[:, 1], 'b', label='Train')
    plt.plot(valid[:, 0], valid[:, 1], 'g', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(fname + '_' + ylabel + '.png')
    plt.clf()


def nn_main(*args):
    # Run NN and save model, stats
    fname = nn_with_args(*args)
    # Plot and save figure based on model, stats
    plot_figures('nn_stats/' + fname + '.npz')


if __name__ == '__main__':
    nn_main([16, 32],  # num_hiddens
            0.01,  # eps
            0.5,  # momentum
            15,  # num_epochs
            100)  # batch_size
