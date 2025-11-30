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
from src.agent import DQNAgent
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

    agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed)

    os.makedirs(save_dir, exist_ok=True)
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            valid_actions = list(G.neighbors(env.current))
            # debug_log("Acting")
            action = agent.act(state, valid_actions)
            # debug_log("Stepping")
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            debug_log(f"Step: ep={ep}, steps={env.steps}")
            agent.replay_train()
            debug_log("Replay done")
            state = next_state
            total_reward += reward
        debug_log(f"Episode {ep} done")

        rewards.append(total_reward)

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_recent = np.mean(rewards[-50:])
            print(f'Ep {ep+1}/{episodes}  TotalReward={total_reward:.2f}  Avg50={avg_recent:.2f}  Eps={agent.epsilon:.3f}')

    model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
    debug_log(f"Saving model to {model_path}")
    try:
        agent.save(model_path)
        debug_log("Model saved successfully")
    except Exception as e:
        debug_log(f"Error saving model: {e}")
        # Try saving weights only as fallback
        weights_path = os.path.join(save_dir, 'dqn_routing_weights.h5')
        debug_log(f"Trying to save weights to {weights_path}")
        agent.model.save_weights(weights_path)

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
    print('Training finished. Model saved to models_test/dqn_routing_tf.keras')
