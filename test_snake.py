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
            predicts = []
            state_infos = []
            # Predict move by current state, angle and all possible moves
            for m in range(-1, 2):
                state_info = s.copy()
                state_info.append(a)
                state_info.append(m)

                x = numpy.array(state_info).reshape(-1, 5, 1)
                predict = model.predict(x)
                # predicts.append(list(predict[0]))
                # Rounded the predict value to 2 decimal places
                predicts.append(list([format(p, '.2f') for p in predict[0]]))

                state_infos.append(state_info)

            predicts_sorted = sorted(predicts, key=operator.itemgetter(0, 1, 2))
            # predicts_sorted = sorted(predicts, key=operator.itemgetter(2), reverse=True)
            move = predicts.index(predicts_sorted[0]) - 1

            # Move the snake
            s, a, d = agent.next_state(move)

            if not agent.alive:
                print(agent.code_id, state_infos[move + 1], agent.score)
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
    predicts = []
    for i in range(-1, 2):
        s = [1, 0, 1, 0.5, i]
        s = numpy.array(s).reshape(-1, 5, 1)

        predict = model.predict(s)

        # predicts.append(list(predict[0]))
        predicts.append(list([format(p, '.2f') for p in predict[0]]))

        print('Move:', i, end=' => ')
        for j in range(len(predict[0])):
            print(format(predict[0][j], '.15f'), predicts[i + 1][j], '|', end=' ')
        print()

    predicts_sorted = sorted(predicts, key=operator.itemgetter(0, 1, 2))
    # predicts_sorted = sorted(predicts, key=operator.itemgetter(2), reverse=True)
    move = predicts.index(predicts_sorted[0]) - 1
    print('selected move:', move)


def main():
    print('--start--')
    # change dir path here
    time_path = '2018-11-13_19-12-57,opt=SGD,lr=0.01,b=128,e=10'
    model = Model.load_model(time_path)

    # test_single(model)
    test_in_game(model, 10, True, True, 30)
    print('--end--')


if __name__ == '__main__':
    main()
