import numpy
from collections import deque
from random import sample

from agent import Agent
from memory import ExperienceReplay


class DQN:
    def __init__(self, model, memory_size, img_size, num_frames, actions):
        self.model = model
        # slow
        # self.memory = deque() if memory_size == -1 else deque(maxlen=memory_size)
        self.memory = ExperienceReplay(memory_size)
        self.img_size = img_size
        self.num_frames = num_frames
        self.actions = actions
        self.num_actions = len(actions)
        self.frames = None

    # slow
    # def record_memory(self, state, action_index, reward, next_state, alive):
    #     self.memory.append((state, action_index, reward, next_state, alive))
    #
    # def recall_memory(self, batch_size):
    #     if len(self.memory) >= batch_size:
    #         return sample(self.memory, batch_size)
    #     else:
    #         return None

    def get_frames(self, board):
        if self.frames is None:
            self.frames = [board] * self.num_frames
        else:
            self.frames.append(board)
            self.frames.pop(0)
        return numpy.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None

    def train(self, episodes, batch_size, gamma, epsilon, epsilon_rate):

        delta = ((epsilon[0] - epsilon[1]) / (episodes * epsilon_rate))
        final_epsilon = epsilon[1]
        epsilon = epsilon[0]

        eat_count = 0

        for e in range(episodes):
            agent = Agent(e, False, False, -1)

            _, _, _, board, reward = agent.get_state()
            self.clear_frames()
            state = self.get_frames(board)

            loss = 0.0

            while agent.alive:
                if numpy.random.random() > epsilon:
                    # use prediction
                    # q_state = self.model.predict(state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                    q_state = self.model.predict(state.reshape(-1, self.num_frames, self.img_size, self.img_size))
                    action_index = int(numpy.argmax(q_state[0]))
                else:
                    # Explore
                    action_index = numpy.random.randint(self.num_actions)

                _, _, _, board, reward = agent.next_state(action_index)

                next_state = self.get_frames(board)
                transition = [state, action_index, reward, next_state, not agent.alive]
                self.memory.remember(*transition)
                # self.record_memory(state, action_index, reward, next_state, agent.alive)
                state = next_state

                batch = self.memory.get_batch(model=self.model, batch_size=batch_size, gamma=gamma)
                if batch:
                    inputs, targets = batch
                    loss += self.model.train_on_batch(inputs, targets)

                # slow
                # batch = self.recall_memory(batch_size)
                # if batch:
                #     x = []
                #     y = []
                #
                #     for state, action_index, reward, next_state, alive in batch:
                #         # q_state = self.model.predict(
                #         #     state.reshape(-1, self.img_size, self.img_size, self.num_frames)).flatten()
                #         q_state = self.model.predict(
                #             state.reshape(-1, self.num_frames, self.img_size, self.img_size)).flatten()
                #         if reward < 0:
                #             # Q[action_index] ~> negative value
                #             q_state[action_index] = reward
                #         else:
                #             # q_next_state = self.model.predict(
                #             #     next_state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                #             q_next_state = self.model.predict(
                #                 next_state.reshape(-1, self.num_frames, self.img_size, self.img_size))
                #             q_state[action_index] = reward + gamma * numpy.max(q_next_state)
                #         x.append(state)
                #         y.append(q_state)
                #
                #     # x = numpy.array(x).reshape(-1, self.img_size, self.img_size, self.num_frames)
                #     x = numpy.array(x).reshape(-1, self.num_frames, self.img_size, self.img_size)
                #     y = numpy.array(y)
                #
                #     loss += self.model.train_on_batch(x, y)

            # TODO - save weight as checkpoint
            # Log
            if agent.score:
                eat_count += 1
            # Tune epsilon
            if epsilon > final_epsilon:
                epsilon -= delta

            # Show result
            print('Episode {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | '
                  'Step {} | Score {} | Eat {}'.format(e + 1, episodes, loss, epsilon, agent.step, agent.score,
                                                       eat_count))

    def test_game(self, model, episodes):
        self.model = model

        # State info
        score_list = []
        step_list = []

        for e in range(episodes):
            agent = Agent(e, False, True, 30)

            _, _, _, board, reward = agent.get_state()
            state = self.get_frames(board)

            while agent.alive:
                # q_state = self.model.predict(state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                q_state = self.model.predict(state.reshape(-1, self.num_frames, self.img_size, self.img_size))
                action_index = int(numpy.argmax(q_state[0]))

                _, _, _, board, reward = agent.next_state(action_index)

                state = self.get_frames(board)

            # Record state
            score_list.append(agent.score)
            step_list.append(agent.step)

        print('Total Games:', episodes)
        print('Total Steps:', sum(step_list))
        print('Avg Steps:', sum(step_list) / float(len(step_list)))
        print('Max Score:', max(score_list))
        print('Avg Score:', sum(score_list) / float(len(score_list)))
