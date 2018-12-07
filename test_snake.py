from dqn import DQN
import model as Model


def main():
    print('--start--')
    # change dir path here
    time_path = 'sb_5'
    model = Model.load_model(time_path)

    # Hyper parameters
    img_size = 12
    num_frames = 4

    actions = [[-1, 0],  # 0 - left
               [0, -1],  # 1 - up
               [1, 0],  # 2 - right
               [0, 1]]  # 3 - down

    # Create DQN Agent
    dqn = DQN(model, model, 0, [-1, -1], img_size, num_frames, actions)

    # Test on game
    dqn.test_game(10, True, 30, True)

    print('--end--')


if __name__ == '__main__':
    main()
