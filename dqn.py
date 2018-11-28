import numpy
from collections import deque
from random import sample

from agent import Agent


class DQN:
    def __init__(self, model, memory_size, img_size, num_frames, actions):
        self.model = model
        self.memory = deque() if memory_size == -1 else deque(maxlen=memory_size)
        self.img_size = img_size
        self.num_frames = num_frames
        self.actions = actions
        self.num_actions = len(actions)

    def record_memory(self, state, action_index, reward, next_state, alive):
        self.memory.append((state, action_index, reward, next_state, alive))

    def recall_memory(self, batch_size):
        if len(self.memory) >= batch_size:
            return sample(self.memory, batch_size)
        else:
            return None

    def train(self, episodes, batch_size, gamma, epsilon):

        for e in range(episodes):
            agent = Agent(e, False, False, -1)

            _, _, _, board, reward = agent.get_state()
            state = numpy.asarray([board] * self.num_frames)

            loss = 0.0

            while agent.alive:
                if numpy.random.random() > epsilon:
                    # use prediction
                    q_state = self.model.predict(state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                    action_index = int(numpy.argmax(q_state[0]))
                    # action_index = numpy.argmax(self.model.predict(state[np.newaxis]), axis=-1)[0]
                else:
                    # Explore
                    action_index = numpy.random.randint(self.num_actions)

                _, _, _, board, reward = agent.next_state(action_index)

                next_state = numpy.roll(state, 1, axis=0)
                next_state[0] = board
                self.record_memory(state, action_index, reward, next_state, agent.alive)
                state = next_state

                batch = self.recall_memory(batch_size)
                if batch:
                    x = []
                    y = []

                    for state, action_index, reward, next_state, alive in batch:
                        q_state = self.model.predict(
                            state.reshape(-1, self.img_size, self.img_size, self.num_frames)).flatten()
                        # q_state = self.model.predict(s.reshape(-1, 4, self.img_size, self.img_size)).flatten()
                        if reward < 0:
                            # Q[action_index] ~> negative value
                            q_state[action_index] = reward
                        else:
                            q_next_state = self.model.predict(
                                next_state.reshape(-1, self.img_size, self.img_size, self.num_frames))
                            # q_next_state = self.model.predict(s_prime[numpy.newaxis]).max(axis=-1)
                            q_state[action_index] = reward + gamma * numpy.max(q_next_state)
                        x.append(state)
                        y.append(q_state)

                    x = numpy.array(x).reshape(-1, self.img_size, self.img_size, self.num_frames)
                    y = numpy.array(y)

                    loss += self.model.train_on_batch(x, y)
                    # TODO - save weight as checkpoint

            # Tune epsilon
            if epsilon > 0.1:
                epsilon -= (0.9 / (episodes / 2.0))

            # Show result
            print(
                'Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Step {} | Score {}'.format(e + 1, episodes, loss,
                                                                                                 epsilon, agent.step,
                                                                                                 agent.score))
