"""Train a DQN agent for routing on a small NetworkX graph.

This script uses `src/env.py` and `src/agent.py`.
"""
import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.env import NetworkRoutingEnv
from src.agent import DQNAgent


def build_sample_graph():
    G = nx.Graph()
    edges = [
        (0, 1, 1.0), (0, 2, 2.5), (1, 2, 1.5),
        (1, 3, 2.0), (2, 4, 1.0), (3, 4, 0.5),
        (3, 5, 3.0), (4, 5, 2.5)
    ]
    G.add_weighted_edges_from(edges)
    return G


def train(episodes=500, save_dir='models', seed=0):
    random.seed(seed)
    np.random.seed(seed)

    G = build_sample_graph()
    env = NetworkRoutingEnv(G, reward_mode='C', seed=seed)
    state_dim = env._get_state().shape[0]
    action_dim = env.num_nodes

    agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed)

    os.makedirs(save_dir, exist_ok=True)
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            valid_actions = list(G.neighbors(env.current))
            action = agent.act(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay_train()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_recent = np.mean(rewards[-50:])
            print(f'Ep {ep+1}/{episodes}  TotalReward={total_reward:.2f}  Avg50={avg_recent:.2f}  Eps={agent.epsilon:.3f}')

    model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
    agent.save(model_path)

    # Plot learning curve
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Learning curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.close()

    return agent, G


if __name__ == '__main__':
    agent, G = train(episodes=300, save_dir='models_demo', seed=0)
    print('Training finished. Model saved to models_demo/dqn_routing_tf.keras')
