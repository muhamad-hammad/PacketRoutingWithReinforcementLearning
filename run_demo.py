"""Small wrapper to train a short demo and run evaluation/plotting."""
import os
import numpy as np

from train_dqn_tf import train
from evaluate import compare_with_dijkstra


def run_demo():
    agent, G = train(episodes=200, save_dir='models_demo', seed=1)
    stats = compare_with_dijkstra(agent, G, trials=30)
    # summarize
    agent_success = np.mean([s['agent_success'] for s in stats])
    dij_success = np.mean([s['dij_success'] for s in stats])
    print(f'Agent success rate: {agent_success:.2f}, Dijkstra success rate: {dij_success:.2f}')


if __name__ == '__main__':
    run_demo()
