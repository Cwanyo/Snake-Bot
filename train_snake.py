import os
import time
import random
import operator
import numpy

import keras

from sklearn.metrics import confusion_matrix, classification_report

from matplotlib import pyplot as plt
import itertools

from agent import Agent
import data as Data
import model as Model
from test_snake import test_in_game


def train(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, log_dir):
    # Use tensorboard
    # cli => tensorboard --logdir files/training_logs
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                              batch_size=batch_size)

    # Train
    # NOTE* - The validation set is checked during training to monitor progress,
    # and possibly for early stopping, but is never used for gradient descent.
    # REF -> https://github.com/keras-team/keras/issues/1753
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_valid, y_valid), callbacks=[tensorboard])


def evaluate(model, movement_classes, state_classes, x_test, y_test, output_dir):
    print('_________________________________________________________________')
    # Evaluate with testing data
    evaluation = model.evaluate(x_test, y_test)

    print('Summary: Loss over the testing dataset: %.2f' % evaluation[0])
    print('- movement_output loss: %.2f' % evaluation[1])
    print('- state_output loss: %.2f' % evaluation[2])
    print('- movement_output accuracy: %.2f' % evaluation[3])
    print('- state_output accuracy: %.2f' % evaluation[4])

    # Get prediction from given x_test
    y_pred = model.predict(x_test)

    k = list(y_test.keys())

    # Get report and confusion matrix for movement
    print('_________________________________________________________________')
    print(k[0])
    movement_cr = classification_report(numpy.argmax(y_test[k[0]], axis=1), numpy.argmax(y_pred[0], axis=1),
                                        target_names=movement_classes)
    print(movement_cr)

    movement_cm = confusion_matrix(numpy.argmax(y_test[k[0]], axis=1), numpy.argmax(y_pred[0], axis=1))
    plot_confusion_matrix(movement_cm, movement_classes)
    # Save evaluate files
    plt.savefig(output_dir + 'movement_confusion_matrix.jpg')
    with open(output_dir + 'movement_confusion_matrix.txt', 'w') as f:
        f.write(numpy.array2string(movement_cm, separator=', '))
    with open(output_dir + 'movement_classification_report.txt', 'w') as f:
        f.write(movement_cr)

    plt.gcf().clear()
    # Get report and confusion matrix for state
    print('_________________________________________________________________')
    print(k[1])
    state_cr = classification_report(numpy.argmax(y_test[k[1]], axis=1), numpy.argmax(y_pred[1], axis=1),
                                     target_names=state_classes)
    print(state_cr)

    state_cm = confusion_matrix(numpy.argmax(y_test[k[1]], axis=1), numpy.argmax(y_pred[1], axis=1))
    plot_confusion_matrix(state_cm, state_classes)
    # Save evaluate files
    plt.savefig(output_dir + 'state_confusion_matrix.jpg')
    with open(output_dir + 'state_movement_confusion_matrix.txt', 'w') as f:
        f.write(numpy.array2string(state_cm, separator=', '))
    with open(output_dir + 'state_movement_classification_report.txt', 'w') as f:
        f.write(state_cr)

    print('=================================================================')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def make_hparam_string(opt, learning_rate, batch_size, epochs):
    # Get current current time
    t = time.strftime('%Y-%m-%d_%H-%M-%S')

    return '%s,opt=%s,lr=%s,b=%d,e=%d/' % (t, opt, learning_rate, batch_size, epochs)


def prepare_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Created directory ' + path)


def main():
    print('--start--')

    # Folder Paths
    log_dir = './files/training_logs/'

    # Hyper parameters
    img_size = 22  # game board size + 2
    num_channels = 1

    movement_classes = ['Move Left', 'Move Up', 'Move Right', 'Move Down']
    state_classes = ['Dead', 'Alive: Wrong Direction', 'Alive: Right Direction']
    num_movement_classes = len(movement_classes)
    num_state_classes = len(state_classes)

    epochs = 10
    batch_size = 128
    learning_rate = 0.01

    # Load Data
    x_train, movement_y_train, state_y_train = Data.generate_data(1000, img_size, num_movement_classes,
                                                                  num_state_classes)
    x_valid, movement_y_valid, state_y_valid = Data.generate_data(200, img_size, num_movement_classes,
                                                                  num_state_classes)

    # Build model
    model = Model.build_model(img_size, num_channels, num_movement_classes, num_state_classes, learning_rate)

    # View model summary
    model.summary()

    # Check memory needed during the training process (not accurate)
    Model.get_model_memory_usage(batch_size, model)

    # Get optimizer name
    opt_name = model.optimizer.__class__.__name__
    # Get folder name
    hparam_str = make_hparam_string(opt_name, learning_rate, batch_size, epochs)
    log_dir += hparam_str
    output_dir = log_dir + 'model/'
    # Create folder
    prepare_dir(output_dir)

    # Train the model
    train(model, x_train, {'movement_output': movement_y_train, 'state_output': state_y_train},
          x_valid, {'movement_output': movement_y_valid, 'state_output': state_y_valid},
          batch_size, epochs, log_dir)

    # Evaluate the model
    evaluate(model, movement_classes, state_classes,
             x_valid, {'movement_output': movement_y_valid, 'state_output': state_y_valid},
             output_dir)

    # Save the model
    Model.save_model(model, movement_classes, state_classes, output_dir)

    # Test on game
    # test_in_game(model, 1000, False, True, 200)

    # Visualize
    # plt.show()

    print('--end--')


if __name__ == '__main__':
    main()
