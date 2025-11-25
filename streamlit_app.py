import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
import tensorflow as tf
from src.env import NetworkRoutingEnv
from src.agent import DQNAgent
import evaluate

# Set page config
st.set_page_config(page_title="Packet Routing RL", layout="wide")

# --- Helper Functions ---

def generate_random_network(num_nodes, density):
    """Generate a random network with weighted edges."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < density:
                weight = random.randint(1, 10)
                G.add_edge(i, j, weight=weight)
    
    # Ensure connectivity
    components = list(nx.connected_components(G))
    while len(components) > 1:
        comp1 = random.choice(list(components[0]))
        comp2 = random.choice(list(components[1]))
        weight = random.randint(1, 10)
        G.add_edge(comp1, comp2, weight=weight)
        components = list(nx.connected_components(G))
    return G

def draw_network(G, pos, path=None, path_color='red', title=None):
    """Draw the network using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Node colors
    node_colors = ['lightblue'] * len(G.nodes())
    if path:
        # Highlight source (green) and current/end (red) if path exists
        if len(path) > 0:
            node_colors[path[0]] = 'green'
            node_colors[path[-1]] = 'red'
            # Highlight intermediate nodes
            for node in path[1:-1]:
                node_colors[node] = 'yellow'

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    # Edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    # Draw path if provided
    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=path_color, width=2.5, ax=ax)

    if title:
        ax.set_title(title)
    ax.axis('off')
    return fig

# --- Session State Initialization ---

if 'graph' not in st.session_state:
    st.session_state.graph = generate_random_network(5, 0.4)
    st.session_state.pos = nx.spring_layout(st.session_state.graph, seed=42)

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'training_history' not in st.session_state:
    st.session_state.training_history = []

# --- Sidebar Controls ---

st.sidebar.title("Configuration")

st.sidebar.subheader("Network Settings")
num_routers = st.sidebar.number_input("Number of Routers", min_value=2, max_value=50, value=5)
density = st.sidebar.slider("Connection Density", 0.1, 1.0, 0.4)

if st.sidebar.button("New Network"):
    st.session_state.graph = generate_random_network(num_routers, density)
    st.session_state.pos = nx.spring_layout(st.session_state.graph, seed=42)
    st.session_state.agent = None # Reset agent on new network
    st.session_state.training_history = []
    st.success("New network generated!")

st.sidebar.divider()

st.sidebar.subheader("Training Params")
episodes = st.sidebar.number_input("Episodes", min_value=10, value=300)
save_dir = st.sidebar.text_input("Save Directory", value="models_demo")
seed = st.sidebar.number_input("Random Seed", value=0)

st.sidebar.divider()

st.sidebar.subheader("Simulation Params")
src_node = st.sidebar.number_input("Source Node", min_value=0, max_value=num_routers-1, value=0)
dst_node = st.sidebar.number_input("Dest Node", min_value=0, max_value=num_routers-1, value=min(4, num_routers-1))
sim_speed = st.sidebar.slider("Simulation Speed (s)", 0.1, 2.0, 0.5)


# --- Main Area ---

st.title("Packet Routing with Reinforcement Learning")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Topology")
    fig = draw_network(st.session_state.graph, st.session_state.pos, title="Current Network")
    st.pyplot(fig)

with col2:
    st.subheader("Network Stats")
    G = st.session_state.graph
    st.write(f"**Nodes:** {G.number_of_nodes()}")
    st.write(f"**Edges:** {G.number_of_edges()}")
    st.write(f"**Avg Degree:** {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    
    if st.session_state.agent:
        st.success("Agent Loaded/Trained")
    else:
        st.warning("No Agent (Train or Load Model)")

# --- Tabs ---

tab_train, tab_sim = st.tabs(["Training", "Simulation & Comparison"])

with tab_train:
    st.header("Train Agent")
    
    if st.button("Start Training"):
        os.makedirs(save_dir, exist_ok=True)
        
        # Reset agent
        G = st.session_state.graph
        env = NetworkRoutingEnv(G, reward_mode='C', seed=seed)
        state_dim = env._get_state().shape[0]
        action_dim = env.num_nodes
        agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        rewards = []
        st.session_state.training_history = []
        
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
            st.session_state.training_history.append(total_reward)
            
            # Update UI every 5 episodes or last one
            if (ep + 1) % 5 == 0 or (ep + 1) == episodes:
                avg50 = sum(rewards[-50:]) / min(len(rewards), 50)
                status_text.text(f"Episode {ep+1}/{episodes} | Avg Reward (last 50): {avg50:.2f} | Epsilon: {agent.epsilon:.3f}")
                progress_bar.progress((ep + 1) / episodes)
                
                # Update chart
                chart_placeholder.line_chart(st.session_state.training_history)
                
                # Force garbage collection to prevent memory leaks
                import gc
                gc.collect()
        
        # Save model
        model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
        agent.save(model_path)
        st.session_state.agent = agent
        st.success(f"Training Complete! Model saved to {model_path}")

    # Load Model
    st.divider()
    st.subheader("Load Existing Model")
    uploaded_file = st.file_uploader("Choose a .keras model file", type="keras")
    if uploaded_file is not None:
        # Save to temp file to load
        with open("temp_model.keras", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            G = st.session_state.graph
            state_dim = NetworkRoutingEnv(G)._get_state().shape[0]
            agent = DQNAgent(state_dim, G.number_of_nodes(), graph=G)
            agent.load("temp_model.keras")
            st.session_state.agent = agent
            st.success("Model Loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

with tab_sim:
    st.header("Simulation")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        if st.button("Simulate Agent"):
            if st.session_state.agent is None:
                st.error("Please train or load an agent first.")
            else:
                agent = st.session_state.agent
                G = st.session_state.graph
                
                # Deterministic for simulation
                original_epsilon = agent.epsilon
                agent.epsilon = 0.0
                
                env = NetworkRoutingEnv(G, reward_mode='C')
                env.reset(src=src_node, dst=dst_node)
                
                path = [src_node]
                placeholder = st.empty()
                
                done = False
                step = 0
                while not done and step < env.max_steps:
                    # Update visualization
                    fig = draw_network(G, st.session_state.pos, path=path, title=f"Step {step}: {path[-1]}")
                    placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(sim_speed)
                    
                    state = env._get_state()
                    neighbors = list(G.neighbors(env.current))
                    
                    # Safety check for stuck agent
                    if not neighbors:
                        st.warning("Dead end reached!")
                        break

                    action = agent.act(state, neighbors)
                    _, _, done, _ = env.step(action)
                    path.append(action)
                    step += 1
                
                # Final state
                fig = draw_network(G, st.session_state.pos, path=path, title=f"Finished: {path[-1]}")
                placeholder.pyplot(fig)
                plt.close(fig)
                
                if path[-1] == dst_node:
                    st.success(f"Reached Destination in {step} steps!")
                else:
                    st.error("Failed to reach destination.")
                
                agent.epsilon = original_epsilon

    with col_sim2:
        if st.button("Compare with Dijkstra"):
            if st.session_state.agent is None:
                st.error("Please train or load an agent first.")
            else:
                G = st.session_state.graph
                agent = st.session_state.agent
                
                # Dijkstra
                try:
                    dijkstra_path = nx.shortest_path(G, src_node, dst_node, weight='weight')
                    dijkstra_cost = nx.shortest_path_length(G, src_node, dst_node, weight='weight')
                except nx.NetworkXNoPath:
                    st.error("No path exists!")
                    dijkstra_path = []
                    dijkstra_cost = float('inf')

                # RL Agent
                original_epsilon = agent.epsilon
                agent.epsilon = 0.0
                
                env = NetworkRoutingEnv(G, reward_mode='C')
                env.reset(src=src_node, dst=dst_node)
                rl_path = [src_node]
                rl_cost = 0
                done = False
                step = 0
                
                while not done and step < env.max_steps:
                    state = env._get_state()
                    neighbors = list(G.neighbors(env.current))
                    action = agent.act(state, neighbors)
                    _, _, done, _ = env.step(action)
                    
                    # Calculate cost
                    if len(rl_path) > 0:
                         rl_cost += G[rl_path[-1]][action]['weight']
                    
                    rl_path.append(action)
                    step += 1
                
                agent.epsilon = original_epsilon
                
                # Display Results
                st.write("### Results")
                
                st.write(f"**Dijkstra Path:** {dijkstra_path}")
                st.write(f"**Cost:** {dijkstra_cost}")
                
                st.write(f"**RL Agent Path:** {rl_path}")
                st.write(f"**Cost:** {rl_cost}")
                
                if rl_cost == dijkstra_cost:
                    st.success("RL Agent found the optimal path! 🎉")
                elif rl_cost < dijkstra_cost:
                    st.warning("RL Agent found a path with lower cost? (Check logic)")
                else:
                    diff = ((rl_cost - dijkstra_cost) / dijkstra_cost) * 100
                    st.info(f"RL Agent path is {diff:.1f}% longer than optimal.")
                
                # Visual Comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot Dijkstra
                nx.draw_networkx_nodes(G, st.session_state.pos, node_color='lightblue', node_size=300, ax=ax1)
                nx.draw_networkx_edges(G, st.session_state.pos, alpha=0.2, ax=ax1)
                if dijkstra_path:
                    path_edges = list(zip(dijkstra_path[:-1], dijkstra_path[1:]))
                    nx.draw_networkx_edges(G, st.session_state.pos, edgelist=path_edges, edge_color='blue', width=2, ax=ax1)
                ax1.set_title("Dijkstra (Optimal)")
                ax1.axis('off')
                
                # Plot RL
                nx.draw_networkx_nodes(G, st.session_state.pos, node_color='lightblue', node_size=300, ax=ax2)
                nx.draw_networkx_edges(G, st.session_state.pos, alpha=0.2, ax=ax2)
                if rl_path:
                    path_edges = list(zip(rl_path[:-1], rl_path[1:]))
                    nx.draw_networkx_edges(G, st.session_state.pos, edgelist=path_edges, edge_color='red', width=2, ax=ax2)
                ax2.set_title("RL Agent")
                ax2.axis('off')
                
                st.pyplot(fig)
