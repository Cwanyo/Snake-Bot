import time
import numpy

from agent import Agent


def generate_data(num_games, num_features, num_classes):
    # Game data
    x_train = []
    y_train = []

    # State info
    score_list = []
    step_list = []
    st = time.time()

    for i in range(num_games):
        agent = Agent(i)
        s, a, d = agent.get_state()

        # TODO - set maximum step due to alive data are more than dead data
        while agent.alive:
            # Generate move
            move = agent.get_random_move(5)

            # Get current state, angle and move
            state_info = s.copy()
            state_info.append(a)
            state_info.append(move)

            # Log state
            pre_distance = agent.food_distance
            pre_score = agent.score

            # Move snake
            s, a, d = agent.next_state(move)

            x_train.append(state_info)
            '''
            Y label
            class 0 = dead
            class 1 = alive but went to the wrong direction
            class 2 = alive and went to the right direction
            '''
            # y_train.append(1 if agent.alive else 0)
            if not agent.alive:
                # Snake is dead
                y_train.append(0)
            else:
                # Snake is alive
                if d < pre_distance or agent.score > pre_score:
                    # Snake went to the right direction
                    y_train.append(2)
                else:
                    # Snake went to the wrong direction
                    y_train.append(1)

        # Record state
        score_list.append(agent.score)
        step_list.append(agent.step)

    # Show state
    print('_________________________________________________________________')
    print('Time:', time.time() - st)
    print('Total Games:', num_games)
    print('Total Steps:', sum(step_list))
    print('Avg Steps:', sum(step_list) / float(len(step_list)))
    print('Max Score:', max(score_list))
    print('Avg Score:', sum(score_list) / float(len(score_list)))

    print('-Class 0:', y_train.count(0))
    print('-Class 1:', y_train.count(1))
    print('-Class 2:', y_train.count(2))

    # Convert y label to onehot encoded
    y_train = onehot(y_train, num_classes)

    x_train = numpy.array(x_train).reshape(-1, num_features, 1)
    y_train = numpy.array(y_train)

    return x_train, y_train


def onehot(y_label, num_classes):
    y_temp = []

    for y in y_label:
        loh = numpy.zeros(num_classes)
        loh[y] = 1.0
        y_temp.append(loh)

    return y_temp
