from util import Load, Save, LoadData
from nn import InitNN, CheckGrad, NNForward, NNBackward, NNUpdate, Train, Evaluate
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


def evaluate_model(model_fname, result_fname, forward, batch_size):
    model = Load(model_fname)
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('../toronto_face.npz')
    train_ce, train_acc = Evaluate(inputs_train, target_train, model, forward, batch_size=batch_size)
    valid_ce, valid_acc = Evaluate(inputs_valid, target_valid, model, forward, batch_size=batch_size)
    test_ce, test_acc = Evaluate(inputs_test, target_test, model, forward, batch_size=batch_size)
    f = open(result_fname, 'w')
    f.write('CE: Train %.5f Validation %.5f Test %.5f\n' % (train_ce, valid_ce, test_ce))
    f.write('Acc: Train {:.5f} Validation {:.5f} Test {:.5f}\n'.format(train_acc, valid_acc, test_acc))
    f.close()
    print('Results written to ' + result_fname)


def plot_first_weights(model_fname):
    model = Load(model_fname)
    W1 = model['W1']
    plt.clf()
    for i in xrange(W1.shape[1]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(W1[:, i].reshape(48, 48), cmap=plt.cm.gray)
    plt.savefig('visualization/nn_first_layer_weights.png')
    print("nn_first_layer_weights.png saved")


def nn_main(*args):
    # Run NN and save model, stats
    nn_with_args(*args)
    # Get file name based on arguments
    fname = str(args[0]) + '_' + str(args[1]) + '_' + str(args[2]) + '_' + str(args[3]) + '_' + str(args[4])
    # Plot and save figure based on model, stats
    plot_figures('nn_stats/' + fname + '.npz')
    # Evaluate model and save results
    evaluate_model('nn_model/' + fname + '.npz', 'nn_result/' + fname, NNForward, args[4])
    # Plot the first layer weights
    # plot_first_weights('nn_model/' + fname + '.npz')


if __name__ == '__main__':
    nn_main([16, 32],  # num_hiddens
            0.01,  # eps
            0.0,  # momentum
            1000,  # num_epochs
            100)  # batch_size
