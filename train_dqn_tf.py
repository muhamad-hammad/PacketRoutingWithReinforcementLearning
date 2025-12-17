"""Train a DQN agent for packet routing."""
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
from src.agent import DQNAgent, AgentManager
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


def train(episodes=500, save_dir='models', seed=0, agent_type='multi-agent', 
          batch_size=32, buffer_size=10000, train_freq=1):
    """Train a DQN agent for packet routing."""
    random.seed(seed)
    np.random.seed(seed)

    G = build_sample_graph()
    env = NetworkRoutingEnv(G, reward_mode='C', seed=seed)
    state_dim = env._get_state().shape[0]
    action_dim = env.num_nodes

    if agent_type == 'universal':
        debug_log("Using Universal DQN Agent")
        agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed,
                        batch_size=batch_size, buffer_size=buffer_size)
        is_multi_agent = False
    else:
        debug_log("Using Multi-Agent Architecture")
        agent = AgentManager(state_dim, action_dim, graph=G, seed=seed,
                           batch_size=batch_size, buffer_size=buffer_size)
        is_multi_agent = True
    
    debug_log(f"Config: Batch={batch_size}, TrainFreq={train_freq}x, Buffer={buffer_size}")

    os.makedirs(save_dir, exist_ok=True)
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            current_node = env.current
            previous_node = env.previous
            valid_actions = list(G.neighbors(current_node))
            
            if is_multi_agent:
                action = agent.act(state, valid_actions, current_node, avoid_node=previous_node)
            else:
                action = agent.act(state, valid_actions, avoid_node=previous_node)
            
            next_state, reward, done, info = env.step(action)
            
            if 'immediate_loop' in info:
                debug_log(f"Immediate loop at episode {ep+1}")
            elif 'excessive_loops' in info:
                debug_log(f"Excessive looping at episode {ep+1}")
            
            if is_multi_agent:
                agent.remember(state, action, reward, next_state, done, current_node)
                for _ in range(train_freq):
                    agent.replay_train(current_node)
            else:
                agent.remember(state, action, reward, next_state, done)
                for _ in range(train_freq):
                    agent.replay_train()
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)

        if (ep + 1) % 50 == 0 or ep == 0:
            avg_recent = np.mean(rewards[-50:])
            eps_str = f"AvgEps={agent.epsilon:.3f}" if is_multi_agent else f"Eps={agent.epsilon:.3f}"
            print(f'Ep {ep+1}/{episodes}  Reward={total_reward:.2f}  Avg50={avg_recent:.2f}  {eps_str}')

    debug_log(f"Saving to {save_dir}")
    try:
        if is_multi_agent:
            agent.save(save_dir)
            debug_log("Multi-agent models saved")
        else:
            model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
            agent.save(model_path)
            debug_log(f"Universal model saved to {model_path}")
    except Exception as e:
        debug_log(f"Error saving: {e}")

    try:
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.title(f'Learning curve ({agent_type})')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
        plt.close()
        debug_log("Learning curve saved")
    except Exception as e:
        debug_log(f"Error plotting: {e}")

    if is_multi_agent:
        try:
            tables = agent.get_all_forwarding_tables()
            md_path = os.path.join(save_dir, 'forwarding_tables.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write('# Forwarding Tables\n\n')
                for node_id, ft in tables.items():
                    f.write(f'## Node {node_id}\n')
                    if ft:
                        f.write('| Source | Destination | Next Hop |\n')
                        f.write('|---|---|---|\n')
                        for (src, dst), nxt in ft.items():
                            f.write(f'| {src} | {dst} | {nxt} |\n')
                    else:
                        f.write('_No entries._\n')
                    f.write('\n')
            debug_log(f"Forwarding tables written")
        except Exception as e:
            debug_log(f"Error writing tables: {e}")
    
    return agent, G


if __name__ == '__main__':
    agent, G = train(episodes=10, save_dir='models_test', seed=0)
    print('Training finished.')
