from dqn import DQN
import model as Model


def main():
    print('--start--')
    # change dir path here
    time_path = 'sb_5'

    checkpoint_step = 3000
    st_checkpoint = 0
    ed_checkpoint = 49000

    # Hyper parameters
    img_size = 12
    num_frames = 4

    actions = [[-1, 0],  # 0 - left
               [0, -1],  # 1 - up
               [1, 0],  # 2 - right
               [0, 1]]  # 3 - down

    while st_checkpoint <= ed_checkpoint:
        # Load model at checkpoint
        model = Model.load_model(time_path, st_checkpoint)

        # Create DQN Agent
        dqn = DQN(model, model, 0, [-1, -1], img_size, num_frames, actions)

        # Test on game
        print('At Checkpoint:', st_checkpoint)
        dqn.test_game(3, True, 60, False)

        st_checkpoint += checkpoint_step

    print('--end--')


if __name__ == '__main__':
    main()
