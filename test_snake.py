import time
import operator
import numpy

from agent import Agent
import model as Model


def test_in_game(model, num_games=1000, log=False, visualization=False, fps=200):
    # State info
    score_list = []
    step_list = []
    st = time.time()

    for i in range(num_games):
        agent = Agent(i, log, visualization, fps)
        s, a, d = agent.get_state()

        while agent.alive:
            # Predict move by current state, angle and all possible moves
            state_info = s.copy()
            state_info.append(a)
            state_info.append(d)

            x = numpy.array(state_info).reshape(-1, 5, 1)
            predict = model.predict(x)

            # Rounded the predict value to 2 decimal places
            predict = list([format(p, '.2f') for p in predict[0]])

            # Get the best move based on prediction
            move = predict.index(max(predict)) - 1

            # Move the snake
            s, a, d = agent.next_state(move)

            if not agent.alive:
                print(agent.code_id, state_info, agent.score)
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
    print('=================================================================')


def test_single(model):
    s = [1, 1, 0, 0.5, 0.3]
    s = numpy.array(s).reshape(-1, 5, 1)

    predict = model.predict(s)
    print(predict)

    # Rounded the predict value to 2 decimal places
    predict = list([format(p, '.2f') for p in predict[0]])
    print(predict)

    move = predict.index(max(predict)) - 1
    print('selected move:', move)


def main():
    print('--start--')
    # change dir path here
    time_path = '2018-11-14_01-07-36,opt=SGD,lr=0.01,b=128,e=10'
    model = Model.load_model(time_path)

    # test_single(model)
    test_in_game(model, 10, True, True, 30)
    print('--end--')


if __name__ == '__main__':
    main()
