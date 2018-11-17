import time
import numpy

from agent import Agent


def generate_data(num_games, img_size, num_classes):
    # Game data
    x_train = []
    y_train = []

    # State info
    score_list = []
    step_list = []
    st = time.time()

    for i in range(num_games):
        agent = Agent(i)
        s, a, d, b = agent.get_state()

        # TODO - set maximum step due to alive data are more than dead data
        while agent.alive:
            # Generate move
            move = agent.get_random_move(0)

            # Get current board
            pre_b = b.copy()

            # Move snake
            s, a, d, b = agent.next_state(move)

            '''
            Y label - use 4 heading directions, not -1 0 1
            class 0 = move left
            class 1 = move up 
            class 2 = move right
            class 3 = move down
            also use alive data only
            '''
            if agent.alive:
                x_train.append(pre_b)
                y_train.append(agent.snake.heading_direction)

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
    print('-Class 3:', y_train.count(3))

    # Convert y label to onehot encoded
    y_train = onehot(y_train, num_classes)

    x_train = numpy.array(x_train).reshape(-1, img_size, img_size, 1)
    y_train = numpy.array(y_train)

    return x_train, y_train


def onehot(y_label, num_classes):
    y_temp = []

    for y in y_label:
        loh = numpy.zeros(num_classes)
        loh[y] = 1.0
        y_temp.append(loh)

    return y_temp
