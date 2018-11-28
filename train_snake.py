import os
import time
import random
import operator
import numpy

import keras

from sklearn.metrics import confusion_matrix, classification_report

from matplotlib import pyplot as plt
import itertools

from dqn import DQN

from agent import Agent
import data as Data
import model as Model
from test_snake import test_in_game


def evaluate(model, classes, x_test, y_test, output_dir):
    print('_________________________________________________________________')
    # Evaluate with testing data
    evaluation = model.evaluate(x_test, y_test)

    print('Summary: Loss over the testing dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

    # Get prediction from given x_test
    y_pred = model.predict_classes(x_test)

    cr = classification_report(numpy.argmax(y_test, axis=1), y_pred, target_names=classes)
    # Get report
    print(cr)

    # Get confusion matrix
    cm = confusion_matrix(numpy.argmax(y_test, axis=1), y_pred)
    plot_confusion_matrix(cm, classes)

    # Save evaluate files
    plt.savefig(output_dir + 'confusion_matrix.jpg')

    with open(output_dir + 'confusion_matrix.txt', 'w') as f:
        f.write(numpy.array2string(cm, separator=', '))

    with open(output_dir + 'classification_report.txt', 'w') as f:
        f.write(cr)

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
    img_size = 12
    num_frames = 4
    actions = [[-1, 0],  # 0 - left
               [0, -1],  # 1 - up
               [1, 0],  # 2 - right
               [0, 1],  # 3 - down
               [0, 0]]  # 4 - idle
    num_classes = len(actions)

    # Number of games / epochs
    episodes = 2000
    # Exploration factor
    epsilon = 1.0
    # Discount factor
    gamma = 0.8
    batch_size = 64
    # -1 is unlimited
    memory_size = -1
    learning_rate = 0.001

    # Build model
    model = Model.build_model(img_size, num_frames, num_classes, learning_rate)

    # View model summary
    model.summary()

    # Create DQN Agent
    dqn = DQN(model, memory_size, img_size, num_frames, actions)

    # Check memory needed during the training process (not accurate)
    Model.get_model_memory_usage(batch_size, model)

    # Get optimizer name
    opt_name = model.optimizer.__class__.__name__
    # Get folder name
    hparam_str = make_hparam_string(opt_name, learning_rate, batch_size, episodes)
    log_dir += hparam_str
    output_dir = log_dir + 'model/'
    # Create folder
    prepare_dir(output_dir)

    # Train the model
    dqn.train(episodes, batch_size, gamma, epsilon)

    # Save the model
    Model.save_model(model, output_dir)

    # Test on game
    # test_in_game(model, 1000, False, True, 200)

    # Visualize
    # plt.show()

    print('--end--')


if __name__ == '__main__':
    main()
