"""Train a DQN agent for routing on a small NetworkX graph.

This script uses `src/env.py` and `src/agent.py`.
"""
import os
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from src.env import NetworkRoutingEnv
from src.agent import AgentManager
try:
    from src.debug_utils import debug_log
except ImportError:
    from debug_utils import debug_log


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

    # Multi-Agent Manager
    agent = AgentManager(state_dim, action_dim, graph=G, seed=seed)

    os.makedirs(save_dir, exist_ok=True)
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            current_node = env.current
            valid_actions = list(G.neighbors(current_node))
            
            # Action selection by the specific node agent
            action = agent.act(state, valid_actions, current_node)
            
            next_state, reward, done, info = env.step(action)
            
            # Store experience in that specific node agent's buffer
            agent.remember(state, action, reward, next_state, done, current_node)
            
            # Train that specific agent
            agent.replay_train(current_node)
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_recent = np.mean(rewards[-50:])
            # agent.epsilon is now the average epsilon across all agents
            print(f'Ep {ep+1}/{episodes}  TotalReward={total_reward:.2f}  Avg50={avg_recent:.2f}  AvgEps={agent.epsilon:.3f}')

    debug_log(f"Saving models to {save_dir}")
    try:
        agent.save(save_dir)
        debug_log("Models saved successfully")
    except Exception as e:
        debug_log(f"Error saving models: {e}")

    # Plot learning curve
    debug_log("Plotting learning curve")
    try:
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.title('Learning curve')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
        plt.close()
        debug_log("Plotting finished")
    except Exception as e:
        debug_log(f"Error plotting: {e}")

    return agent, G


if __name__ == '__main__':
    agent, G = train(episodes=10, save_dir='models_test', seed=0)
    print('Training finished. Models saved to models_test/')

