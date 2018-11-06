import time
import numpy

from agent import Agent


def generate_data(num_games=1000):
    # Game data
    x_train = []
    y_train = []

    # State info
    score_list = []
    total_steps = 0
    st = time.time()

    for i in range(num_games):
        agent = Agent(i)
        s, a = agent.get_state()

        # TODO - set maximum step due to alive data are more than dead data
        while agent.alive:
            # Generate move
            move = agent.generate_move()

            # Get current state, angle and move
            state_info = s
            state_info.append(a)
            state_info.append(move)

            s, a = agent.next_state(move)

            x_train.append(state_info)
            '''
            Y label
            class 0 = dead
            class 1 = alive
            '''
            y_train.append(1 if agent.alive else 0)

        # Record state
        score_list.append(agent.score)
        total_steps += agent.step

    # Convert y label to onehot encoded
    y_train = onehot(y_train, 2)

    x_train = numpy.array(x_train).reshape(-1, 5, 1)
    y_train = numpy.array(y_train)

    # Show state
    print('_________________________________________________________________')
    print("Time:", time.time() - st)
    print("Total games:", num_games)
    print("Total steps:", total_steps)
    print("Max score:", max(score_list))
    print("Average score:", sum(score_list) / float(len(score_list)))
    print('=================================================================')

    return x_train, y_train


def onehot(y_label, num_classes):
    y_temp = []

    for y in y_label:
        loh = numpy.zeros(num_classes)
        loh[y] = 1.0
        y_temp.append(loh)

    return y_temp
