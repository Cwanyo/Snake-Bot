import time
import numpy

from agent import Agent


def generate_data(num_games, img_size, num_movement_classes, num_state_classes):
    # Game data
    x_train = []
    movement_y_train = []
    state_y_train = []

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
            move = agent.get_random_move(5)

            # Get current board
            x_train.append(b)

            # Log state
            pre_distance = agent.food_distance
            pre_score = agent.score

            # Move snake
            s, a, d, b = agent.next_state(move)

            '''
            Movement Y label - use 4 heading directions, not -1 0 1
            class 0 = move left
            class 1 = move up 
            class 2 = move right
            class 3 = move down
            also use alive data only
            '''
            movement_y_train.append(agent.snake.heading_direction)

            '''
            State Y label
            class 0 = dead
            class 1 = alive but went to the wrong direction
            class 2 = alive and went to the right direction
            '''
            # y_train.append(1 if agent.alive else 0)
            if not agent.alive:
                # Snake is dead
                state_y_train.append(0)
            else:
                # Snake is alive
                if d < pre_distance or agent.score > pre_score:
                    # Snake went to the right direction
                    state_y_train.append(2)
                else:
                    # Snake went to the wrong direction
                    state_y_train.append(1)

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

    for i in range(num_movement_classes):
        print('-Movement Class {}: {}'.format(i,movement_y_train.count(i)))

    for i in range(num_state_classes):
        print('-State Class {}: {}'.format(i, state_y_train.count(i)))

    # Convert y label to onehot encoded
    movement_y_train = onehot(movement_y_train, num_movement_classes)
    state_y_train = onehot(state_y_train, num_state_classes)

    x_train = numpy.array(x_train).reshape(-1, img_size, img_size, 1)
    movement_y_train = numpy.array(movement_y_train)
    state_y_train = numpy.array(state_y_train)

    return x_train, movement_y_train, state_y_train


def onehot(y_label, num_classes):
    y_temp = []

    for y in y_label:
        loh = numpy.zeros(num_classes)
        loh[y] = 1.0
        y_temp.append(loh)

    return y_temp
