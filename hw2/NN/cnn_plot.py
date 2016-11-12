from util import Save, Load
from nn import CheckGrad, Train
from cnn import InitCNN, CNNForward, CNNBackward, CNNUpdate
from nn_plot import save_figure, evaluate_model
import numpy as np
import matplotlib.pyplot as plt

# Turn interactive plotting off
plt.ioff()


def cnn_with_args(eps=0.1,
                  momentum=0.5,
                  num_epochs=30,
                  filter_size=5,
                  num_filters_1=8,
                  num_filters_2=16,
                  batch_size=100):
    # Input-output dimensions.
    num_channels = 1
    num_outputs = 7

    # Initialize model.
    model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2, num_outputs)

    # Check gradient implementation.
    print('Checking gradients...')
    x = np.random.rand(10, 48, 48, 1) * 0.1
    CheckGrad(model, CNNForward, CNNBackward, 'W3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W1', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b1', x)

    # Train model.
    model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps, momentum, num_epochs, batch_size)

    # Construct file name
    fname = str(eps) + '_' + str(momentum) + '_' + str(num_epochs) + '_' + str(filter_size) + '_' + \
            str(num_filters_1) + '_' + str(num_filters_2) + '_' + str(batch_size)

    # Save the training statistics.
    Save('cnn_model/' + fname, model)
    Save('cnn_stats/' + fname, stats)

    return fname


def plot_figures(stats_fname):
    stats = Load(stats_fname)
    train_ce_list = stats['train_ce']
    valid_ce_list = stats['valid_ce']
    train_acc_list = stats['train_acc']
    valid_acc_list = stats['valid_acc']
    figure_path = get_cnn_path(stats_fname)
    save_figure(train_ce_list, valid_ce_list, 'Cross Entropy', figure_path)
    save_figure(train_acc_list, valid_acc_list, 'Accuracy', figure_path)
    print('Figures saved to ' + figure_path)


def get_cnn_path(stats_fname):
    sep = stats_fname.find('/')
    figure_name = stats_fname[sep + 1:-4]
    return "cnn_figure/" + figure_name


def cnn_main(*args):
    # Run CNN and save model, stats
    fname = cnn_with_args(*args)
    # Plot and save figure based on model, stats
    plot_figures('cnn_stats/' + fname + '.npz')
    # Evaluate model and save results
    evaluate_model('cnn_model/' + fname + '.npz', 'cnn_result/' + fname, CNNForward, args[6])


if __name__ == '__main__':
    cnn_main(0.1,  # eps
             0.5,  # momentum
             2,  # num_epochs
             5,  # filter_size
             8,  # num_filters_1
             16,  # num_filters_1
             100)  # batch_size
