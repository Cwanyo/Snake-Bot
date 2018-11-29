import time
import operator
import numpy

from dqn import DQN
import model as Model


def main():
    print('--start--')
    # change dir path here
    time_path = '2018-11-29_04-33-19,opt=RMSprop,lr=0.001,b=64,e=1000'
    model = Model.load_model(time_path)

    # Hyper parameters
    img_size = 22
    num_frames = 4
    actions = [[-1, 0],  # 0 - left
               [0, -1],  # 1 - up
               [1, 0],  # 2 - right
               [0, 1],  # 3 - down
               [0, 0]]  # 4 - idle

    # -1 is unlimited
    memory_size = -1

    # Create DQN Agent
    dqn = DQN(model, memory_size, img_size, num_frames, actions)

    # Test on game
    dqn.test_game(model, 1000)

    print('--end--')


if __name__ == '__main__':
    main()
