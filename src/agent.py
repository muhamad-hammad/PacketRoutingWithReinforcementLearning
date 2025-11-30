import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
try:
    from src.debug_utils import debug_log
except ImportError:
    from debug_utils import debug_log


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

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
            debug_log("Random action")
            return random.choice(valid_actions)
        q = self.model(state.reshape(1, -1), training=False).numpy()[0]
        # pick valid action with max q
        return max(valid_actions, key=lambda a: q[a])

    def remember(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns, d)

    def replay_train(self):
        if len(self.replay) < self.batch_size:
            return 0
        
        debug_log("Starting replay_train")

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        
        # Predict Q-values for current states and next states
        targets = self.model(states, training=False).numpy()
        next_qs = self.target(next_states, training=False).numpy()

        # Vectorized target calculation
        # If done, target is reward. Else reward + gamma * max(next_q)
        
        # We need to handle the graph-constrained next_max carefully.
        # If we don't have the graph, we just take the max over all actions.
        if self.graph is None:
            next_max = np.max(next_qs, axis=1)
        else:
            # For graph-constrained, we need to mask invalid actions for each item in batch.
            # This is hard to fully vectorize without an adjacency matrix.
            # We'll do a semi-vectorized approach: pre-calculate maxes row-by-row only if needed.
            
            # Optimization: If the graph is static, we could cache adjacency.
            # For now, we'll keep the logic but optimize the loop.
            
            next_max = np.zeros(self.batch_size)
            num_nodes = self.graph.number_of_nodes()
            
            # Extract next current nodes from one-hot encoding (first num_nodes elements)
            next_cur_nodes = np.argmax(next_states[:, :num_nodes], axis=1)
            
            for i in range(self.batch_size):
                if dones[i]:
                    continue
                
                next_cur = next_cur_nodes[i]
                neighbors = list(self.graph.neighbors(next_cur))
                
                if not neighbors:
                    next_max[i] = np.max(next_qs[i]) # Fallback if no neighbors (dead end)
                else:
                    next_max[i] = np.max(next_qs[i][neighbors])

        # Calculate target Q-values
        # target = reward + gamma * next_max * (1 - done)
        target_values = rewards + self.gamma * next_max * (1 - dones.astype(float))
        
        # Update the specific action indices in the targets array
        # We use np.arange to select the row indices and actions for column indices
        batch_indices = np.arange(self.batch_size)
        targets[batch_indices, actions.astype(int)] = target_values

        debug_log("Training on batch")
        try:
            self.model.train_on_batch(states, targets)
            debug_log("Finished train_on_batch")
        except Exception as e:
            debug_log(f"Error in train_on_batch: {e}")
            raise e

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
