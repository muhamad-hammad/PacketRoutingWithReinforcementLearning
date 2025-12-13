import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
# Enable mixed precision if a GPU is available (helps speed up training)
mixed_precision.set_global_policy('mixed_float16')
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



class NodeAgent:
    def __init__(self, state_dim, action_dim, node_id, graph=None, lr=1e-3, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, buffer_size=10000,
                 batch_size=32, target_update_freq=500, seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_id = node_id
        self.graph = graph
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0
        # Forwarding table: (src, dst) -> next_hop
        self.forwarding_table = {}

        if seed is not None:
            # unique seed per agent
            random.seed(seed + node_id)
            np.random.seed(seed + node_id)
            tf.random.set_seed(seed + node_id)

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
        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        # pick valid action with max q
        return max(valid_actions, key=lambda a: q[a])

    def remember(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns, d)

    def replay_train(self):
        if len(self.replay) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        targets = self.model.predict(states, verbose=0)
        next_qs = self.target.predict(next_states, verbose=0)

        # For more accurate targets, mask the next-state Q-values to only valid neighbor actions
        # (requires agent to have access to the graph). The state layout is expected to
        # have the current-node one-hot in the first num_nodes entries.
        for i in range(len(states)):
            a = int(actions[i])
            if dones[i]:
                targets[i][a] = rewards[i]
            else:
                if self.graph is None:
                    # fallback: take max over all outputs
                    next_max = np.max(next_qs[i])
                else:
                    # infer next current node from the next_state one-hot prefix
                    num_nodes = self.graph.number_of_nodes()
                    next_state = next_states[i]
                    next_cur = int(np.argmax(next_state[:num_nodes]))
                    neighbors = list(self.graph.neighbors(next_cur))
                    if len(neighbors) == 0:
                        next_max = np.max(next_qs[i])
                    else:
                        next_max = np.max(next_qs[i][neighbors])

                targets[i][a] = rewards[i] + self.gamma * next_max

        self.model.fit(states, targets, epochs=1, verbose=0)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target()

        return 1

    def save(self, path):
        # We'll save with a suffix for the node_id in the manager, 
        # but here we just accept a full path
        self.model.save(path)

    # ---------------------------------------------------------------------
    # Forwarding‑table helpers
    # ---------------------------------------------------------------------
    def update_forwarding(self, src, dst, next_hop):
        """Record the best next hop for a (src, dst) pair.
        Called during evaluation or after a training step when the action
        chosen by the policy is considered the current best.
        """
        self.forwarding_table[(src, dst)] = next_hop

    def get_forwarding_table(self):
        """Return the stored forwarding table (a dict)."""
        return self.forwarding_table

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target()


class AgentManager:
    """
    Manages a collection of NodeAgents, one for each node in the graph.
    """
    def __init__(self, state_dim, action_dim, graph, seed=0, **kwargs):
        self.agents = {}
        self.graph = graph
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        nodes = list(graph.nodes())
        for node in nodes:
            self.agents[node] = NodeAgent(
                state_dim, action_dim, node_id=node, graph=graph, seed=seed, **kwargs
            )

    def act(self, state, valid_actions, current_node):
        return self.agents[current_node].act(state, valid_actions)

    def remember(self, s, a, r, ns, d, current_node):
        self.agents[current_node].remember(s, a, r, ns, d)

    def replay_train(self, current_node=None):
        # We can choose to train all agents or just the one active
        # For simplicity, let's train ONLY the active agent per step
        if current_node is not None:
             return self.agents[current_node].replay_train()
        
        # Or if no node specified, train all (heavy!)
        total_loss = 0
        for agent in self.agents.values():
            total_loss += agent.replay_train()
        return total_loss

    @property
    def epsilon(self):
        # Return average epsilon for logging
        epsilons = [a.epsilon for a in self.agents.values()]
        return np.mean(epsilons) if epsilons else 1.0

    @property
    def epsilon_min(self):
        # Just return one of them or min
        return 0.05

    @epsilon.setter
    def epsilon(self, value):
        # Force set epsilon for all agents (e.g. for evaluation)
        for agent in self.agents.values():
            agent.epsilon = value

    def save(self, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        for node, agent in self.agents.items():
            path = os.path.join(base_dir, f"agent_node_{node}.keras")
            agent.save(path)

    def load(self, base_dir):
        for node, agent in self.agents.items():
            path = os.path.join(base_dir, f"agent_node_{node}.keras")
            if os.path.exists(path):
                agent.load(path)
            else:
                print(f"Warning: No model found for node {node} at {path}")

    # ---------------------------------------------------------------------
    # Forwarding‑table aggregation
    # ---------------------------------------------------------------------
    def get_all_forwarding_tables(self):
        """Return a dict mapping node id → its forwarding table dict.
        Useful for inspection after training/evaluation.
        """
        return {node: agent.get_forwarding_table() for node, agent in self.agents.items()}

