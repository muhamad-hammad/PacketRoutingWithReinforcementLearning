import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.env import NetworkRoutingEnv


def extract_route_from_agent(agent, G, src, dst, max_steps=None):
    # Save original epsilon and set to 0 for deterministic evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
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
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
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


def plot_route_comparison(G, route_agent, route_dijkstra, title="Route Comparison"):
    """Plot agent's route vs Dijkstra's shortest path."""
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Plot graph structure
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    
    # Plot edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    # Plot agent's route
    if route_agent:
        path_edges = list(zip(route_agent[:-1], route_agent[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2, label='Agent')
    
    # Plot Dijkstra's route
    if route_dijkstra:
        path_edges = list(zip(route_dijkstra[:-1], route_dijkstra[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='g', width=2, alpha=0.5, label='Dijkstra')
    
    plt.title(title)
    plt.legend()
    return plt.gcf()

def analyze_stats(stats):
    """Analyze and print detailed comparison statistics."""
    n_trials = len(stats)
    n_agent_success = sum(s['agent_success'] for s in stats)
    n_dij_success = sum(s['dij_success'] for s in stats)
    
    # Filter successful cases for both
    successful = [s for s in stats if s['agent_success'] and s['dij_success']]
    
    if successful:
        cost_ratios = [s['agent_cost'] / s['dij_cost'] for s in successful]
        hop_ratios = [s['agent_len'] / max(1, s['dij_len']) for s in successful]
        
        print(f"\nEvaluation Results ({n_trials} trials):")
        print(f"Agent Success Rate: {n_agent_success/n_trials*100:.1f}%")
        print(f"Average Cost Ratio (Agent/Dijkstra): {np.mean(cost_ratios):.2f}")
        print(f"Average Hop Ratio (Agent/Dijkstra): {np.mean(hop_ratios):.2f}")
        print(f"\nDetailed Stats (successful routes only):")
        print(f"Cost Ratio - Mean: {np.mean(cost_ratios):.2f}, Std: {np.std(cost_ratios):.2f}")
        print(f"Hop Ratio  - Mean: {np.mean(hop_ratios):.2f}, Std: {np.std(hop_ratios):.2f}")
    else:
        print("No successful comparisons available.")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.agent import DQNAgent
    from train_dqn_tf import build_sample_graph
    
    # Load trained model
    model_path = 'models_demo/dqn_routing_tf.keras'
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}. Please run training first.")
        exit(1)
    
    # Initialize environment and agent
    G = build_sample_graph()
    state_dim = NetworkRoutingEnv(G)._get_state().shape[0]
    agent = DQNAgent(state_dim, G.number_of_nodes(), graph=G)
    agent.load(model_path)
    
    # Run evaluation
    print("Running evaluation...")
    stats = compare_with_dijkstra(agent, G, trials=50)
    analyze_stats(stats)
    
    # Plot a few example routes
    nodes = list(G.nodes())
    for _ in range(3):  # Plot 3 example routes
        src, dst = np.random.choice(nodes, size=2, replace=False)
        route_agent, _ = extract_route_from_agent(agent, G, src, dst)
        try:
            route_dijkstra = nx.shortest_path(G, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            route_dijkstra = None
        
        fig = plot_route_comparison(G, route_agent, route_dijkstra,
                                  f"Route from {src} to {dst}")
        fig.savefig(f'models_demo/route_comparison_{src}to{dst}.png')
        plt.close(fig)
    
    print("\nRoute visualizations saved to models_demo/route_comparison_*.png")
