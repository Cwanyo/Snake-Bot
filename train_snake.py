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

import model as Model

from keras import backend as K

# (num_frames,img_size,img_size) format
K.set_image_dim_ordering('th')


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
    # img_size = board_size + 2 , where 2 is the border padding
    # TODO - change snake game board setting
    img_size = 10 + 2
    num_frames = 4
    actions = [[-1, 0],  # 0 - left
               [0, -1],  # 1 - up
               [1, 0],  # 2 - right
               [0, 1],  # 3 - down
               [0, 0]]  # 4 - idle
    num_classes = len(actions)

    # Number of games / epochs
    episodes = 50000
    # Exploration factor
    epsilon = [1.0, 0.1]
    epsilon_rate = 0.7
    # Discount factor
    gamma = 0.8
    batch_size = 256
    # -1 is unlimited
    memory_size = 1000000
    learning_rate = 0.001

    # Build model
    model = Model.build_model(img_size, num_frames, num_classes, learning_rate)

    # View model summary
    model.summary()

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

    # Create DQN Agent
    dqn = DQN(model, memory_size, img_size, num_frames, actions, log_dir)

    # Train the model
    dqn.train(episodes, batch_size, gamma, epsilon, epsilon_rate)

    # Save model
    Model.save_model(model, output_dir)

    # Test on game
    # dqn.test_game(model, 1000)

    print('--end--')


if __name__ == '__main__':
    main()
