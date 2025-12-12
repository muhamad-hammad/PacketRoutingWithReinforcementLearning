import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = int(capacity)
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


def build_model(input_dim, output_dim, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim)
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')
    return model


class DQNAgent:
    def __init__(self, state_dim, action_dim, graph=None, lr=1e-3, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, buffer_size=10000,
                 batch_size=32, target_update_freq=500, seed=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.graph = graph
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.model = build_model(state_dim, action_dim, lr=lr)
        self.target = build_model(state_dim, action_dim, lr=lr)
        self.update_target()

        self.replay = ReplayBuffer(capacity=buffer_size)

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def act(self, state, valid_actions):
        # valid_actions: list of neighbor node indices
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        q = self.model(state.reshape(1, -1), training=False).numpy()[0]

        # pick valid action with max q
        return max(valid_actions, key=lambda a: q[a])

    def remember(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns, d)

    def replay_train(self):
        if len(self.replay) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        targets = self.model(states, training=False).numpy()
        next_qs = self.target(next_states, training=False).numpy()

        # For more accurate targets, mask the next-state Q-values to only valid neighbor actions
        # (requires agent to have access to the graph). The state layout is expected to
        # have the current-node one-hot in the first num_nodes entries.
        if self.graph is None:
            # Fallback for when graph is not known (standard DQN)
            next_qs_main_model = self.model(next_states, training=False).numpy()
            best_actions = np.argmax(next_qs_main_model, axis=1)
            next_qs_target = self.target(next_states, training=False).numpy()
            next_max = next_qs_target[np.arange(len(states)), best_actions]

        else:
            # Graph-aware Double DQN
            next_qs_main = self.model(next_states, training=False).numpy()
            next_qs_target = self.target(next_states, training=False).numpy()

            next_max_batch = []
            num_nodes = self.graph.number_of_nodes()

            for i in range(len(states)):
                if dones[i]:
                    next_max_batch.append(0.0)  # Value doesn't matter as it's multiplied by 0 (or not added)
                    continue

                # Infer next current node from the next_state one-hot prefix
                next_state_vec = next_states[i]
                next_cur = int(np.argmax(next_state_vec[:num_nodes]))
                neighbors = list(self.graph.neighbors(next_cur))

                if not neighbors:
                    # Dead end, value is effectively 0 for next step (or penalty already received)
                    next_max_batch.append(0.0)
                    continue

                # Double DQN:
                # 1. Select best action using MAIN model, restricted to valid neighbors
                q_main_neighbors = next_qs_main[i][neighbors]
                # Argmax returns index within the 'neighbors' list
                best_neighbor_idx = np.argmax(q_main_neighbors)
                best_action = neighbors[best_neighbor_idx]

                # 2. Evaluate that action using TARGET model
                next_max_batch.append(next_qs_target[i][best_action])

            next_max = np.array(next_max_batch)

        # Vectorized target calculation where possible, but here we iterated for graph logic
        # If we didn't iterate, we would do:
        # targets[range(len(states)), actions.astype(int)] = rewards + self.gamma * next_max * (1 - dones)

        # Applying the updates to the targets array
        for i in range(len(states)):
            a = int(actions[i])
            if dones[i]:
                targets[i][a] = rewards[i]
            else:
                targets[i][a] = rewards[i] + self.gamma * next_max[i]

        self.model.train_on_batch(states, targets)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target()

        return 1

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target()