"""
Train a DQN agent for routing on a small NetworkX graph.
This script uses src/env.py and src/agent.py.
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
        (0, 1, 1.0),
        (0, 2, 2.5),
        (1, 2, 1.5),
        (1, 3, 2.0),
        (2, 4, 1.0),
        (3, 4, 0.5),
        (3, 5, 3.0),
        (4, 5, 2.5),
    ]

    G.add_weighted_edges_from(edges)
    return G


def train(episodes=500, save_dir="models", seed=0):
    random.seed(seed)
    np.random.seed(seed)

    G = build_sample_graph()
    env = NetworkRoutingEnv(G, reward_mode="C", seed=seed)

    state_dim = env._get_state().shape[0]
    action_dim = env.num_nodes

    agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed)

    os.makedirs(save_dir, exist_ok=True)

    rewards = []

    # --------------------------------------------------
    # Dynamic epsilon decay
    # --------------------------------------------------
    target_decay_episodes = 0.8 * episodes
    if target_decay_episodes < 1:
        target_decay_episodes = 1

    calculated_decay = np.exp(
        np.log(agent.epsilon_min / agent.epsilon) / target_decay_episodes
    )
    agent.epsilon_decay = calculated_decay

    print(f"Training with Dynamic Decay: {agent.epsilon_decay:.4f}")

    # --------------------------------------------------
    # Warmup Phase
    # --------------------------------------------------
    warmup_steps = 1000
    print("Warmup Phase...")

    w_step = 0
    while w_step < warmup_steps:
        w_state = env.reset()
        w_done = False

        while not w_done and w_step < warmup_steps:
            w_valid_actions = list(G.neighbors(env.current))
            w_action = random.choice(w_valid_actions)

            w_next_state, w_reward, w_done, _ = env.step(w_action)
            agent.remember(
                w_state,
                w_action,
                w_reward,
                w_next_state,
                w_done,
            )

            w_state = w_next_state
            w_step += 1

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
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
            print(
                f"Ep {ep + 1}/{episodes}  "
                f"TotalReward={total_reward:.2f}  "
                f"Avg50={avg_recent:.2f}  "
                f"Eps={agent.epsilon:.3f}"
            )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    model_path = os.path.join(save_dir, "dqn_routing_tf.keras")
    agent.save(model_path)

    # --------------------------------------------------
    # Plot learning curve
    # --------------------------------------------------
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Learning curve")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "learning_curve.png"))
    plt.close()

    return agent, G


if _name_ == "_main_":
    agent, G = train(episodes=300, save_dir="models_demo", seed=0)
    print("Training finished. Model saved to models_demo/dqn_routing_tf.keras")
