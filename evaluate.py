import os
import numpy as np
import networkx as nx
from src.env import NetworkRoutingEnv


def extract_route_from_agent(agent, G, src, dst, max_steps=None):
    env = NetworkRoutingEnv(G, reward_mode='C')
    env.reset(src=src, dst=dst)
    path = [src]
    total_cost = 0.0
    max_steps = max_steps or env.max_steps

    for _ in range(max_steps):
        neighbors = list(G.neighbors(env.current))
        if not neighbors:
            break
        state = env._get_state()
        action = agent.act(state, neighbors)
        _, reward, done, _ = env.step(action)
        path.append(action)
        if reward < 0:
            total_cost += -reward
        if done:
            break

    return path, total_cost


def compare_with_dijkstra(agent, G, trials=50):
    nodes = list(G.nodes())
    stats = []
    for _ in range(trials):
        src, dst = np.random.choice(nodes, size=2, replace=False)
        # agent route
        route_agent, cost_agent = extract_route_from_agent(agent, G, src, dst)
        success_agent = route_agent[-1] == dst

        # dijkstra
        try:
            path_dij = nx.shortest_path(G, src, dst, weight='weight')
            cost_dij = sum(G[u][v].get('weight', 1.0) for u, v in zip(path_dij[:-1], path_dij[1:]))
            success_dij = True
        except nx.NetworkXNoPath:
            path_dij = []
            cost_dij = float('inf')
            success_dij = False

        stats.append({
            'src': src, 'dst': dst,
            'agent_len': len(route_agent)-1, 'agent_cost': cost_agent, 'agent_success': success_agent,
            'dij_len': len(path_dij)-1, 'dij_cost': cost_dij, 'dij_success': success_dij
        })

    return stats


if __name__ == '__main__':
    print('Run training first (see train_dqn_tf.py) and pass an agent object to compare_with_dijkstra.')
