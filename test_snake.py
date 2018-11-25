import time
import operator
import numpy

from agent import Agent
import model as Model


def test_in_game(model, num_games=1000, log=False, visualization=False, fps=200):
    # State info
    bad_moves = 0
    score_list = []
    step_list = []
    st = time.time()

    for i in range(num_games):
        mem_step = []

        agent = Agent(i, log, visualization, fps)
        s, a, d, b = agent.get_state()

        while agent.alive:
            # Predict move by current board
            # mem_step.append(b)
            #
            # pre_step = mem_step[len(mem_step) - 100:]
            #
            # board = numpy.array(pre_step).reshape(-1, len(pre_step), 22, 22, 1)
            board = numpy.array(b).reshape(-1, 1, 22, 22, 1)

            predict = model.predict(board)

            predict = numpy.array(predict).reshape(-1, 4)

            # Rounded the predict value to 2 decimal places
            predict = list([format(p, '.2f') for p in predict[-1]])

            # Get the best heading direction based on prediction
            '''
            0 = move left
            1 = move up 
            2 = move right
            3 = move down
            '''
            # Convert to -1 0 1 move
            predict_heading = predict.index(max(predict))
            curr_heading = agent.snake.heading_direction
            invalid_heading = (curr_heading + 2) % 4
            move = 0
            if predict_heading != invalid_heading:
                move = predict_heading - curr_heading

            pre_s = s
            pre_b = b

            # Move the snake
            s, a, d, b = agent.next_state(move)

            if not agent.alive:
                '''
                There are some bad move made by the model. 
                It move to the direction of the food without caring about the obstacles and die.
                pre_s = [1,1,1] is worst case that surrounded by obstacles
                '''
                if sum(pre_s) != 3:
                    bad_moves += 1

                for r in pre_b:
                    for c in r:
                        print(c, end='\t')
                    print()
                print(agent.code_id, pre_s, predict_heading)
                # input('PRESS ENTER TO CON')

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
    print('Bad Moves:', bad_moves)
    print('=================================================================')


def get_board_sample(window_width, window_height):
    snake_head = [19.0, 1.0]
    snake_body = [[19.0, 2.0], [19.0, 3.0], [19.0, 4.0], [19.0, 5.0], [18.0, 5.0], [18.0, 6.0], [17.0, 6.0],
                  [17.0, 7.0], [16.0, 7.0], [16.0, 8.0], [15.0, 8.0]]
    food_location = [18, 1]

    # coordinate x,y are opposite in array => y,x
    temp_board = [[0] * (window_width + 2) for i in range(window_height + 2)]

    # mark top & bottom wall
    for i in range(len(temp_board[0])):
        temp_board[0][i] = -1
        temp_board[len(temp_board) - 1][i] = -1

    # mark left and right wall
    for i in range(len(temp_board)):
        temp_board[i][0] = -1
        temp_board[i][len(temp_board[0]) - 1] = -1

    # mark snake
    temp_board[int(snake_head[1]) + 1][int(snake_head[0]) + 1] = 0.5
    for b in snake_body:
        temp_board[int(b[1]) + 1][int(b[0]) + 1] = -1

    # mark food
    temp_board[int(food_location[1]) + 1][int(food_location[0]) + 1] = 1

    return temp_board


def test_single(model):
    s = get_board_sample(20, 20)

    for r in s:
        for c in r:
            print(c, end='\t')
        print()

    s = numpy.array(s).reshape(-1, 2, 22, 22, 1)

    predict = model.predict(s)
    print(predict)

    predict = numpy.array(predict).reshape(-1, 4)

    # Rounded the predict value to 2 decimal places
    predict = list([format(p, '.2f') for p in predict[0]])
    print(predict)

    move = predict.index(max(predict))
    print('selected move:', move)


def main():
    print('--start--')
    # change dir path here
    time_path = '2018-11-23_03-49-01,opt=SGD,lr=0.01,b=16,e=10'
    model = Model.load_model(time_path)

    # test_single(model)
    test_in_game(model, 10, True, True, 30)
    print('--end--')


if __name__ == '__main__':
    main()
