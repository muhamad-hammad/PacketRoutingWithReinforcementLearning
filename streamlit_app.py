import os
import time
import random
import numpy as np
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf

from src.env import NetworkRoutingEnv
from src.agent import DQNAgent, AgentManager
import evaluate


# ======================================================
# Streamlit Page Configuration
# ======================================================
st.set_page_config(
    page_title="Packet Routing RL",
    layout="wide"
)


# ======================================================
# Helper Functions
# ======================================================
def generate_random_network(num_nodes: int, density: float) -> nx.Graph:
    """Generate a random connected network with weighted edges."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < density:
                G.add_edge(i, j, weight=random.randint(1, 10))

    # Ensure graph connectivity
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1 = random.choice(list(components[0]))
        c2 = random.choice(list(components[1]))
        G.add_edge(c1, c2, weight=random.randint(1, 10))
        components = list(nx.connected_components(G))

    return G


def draw_network(
    G: nx.Graph,
    pos: dict,
    path: list | None = None,
    path_color: str = "red",
    title: str | None = None,
):
    """Draw the network using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 6))

    node_colors = ["lightblue"] * len(G.nodes)

    if path:
        node_colors[path[0]] = "green"
        node_colors[path[-1]] = "red"
        for node in path[1:-1]:
            node_colors[node] = "yellow"

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            edge_color=path_color,
            width=2.5,
            ax=ax,
        )

    if title:
        ax.set_title(title)

    ax.axis("off")
    return fig


def apply_congestion(
    G: nx.Graph,
    congestion_prob: float = 0.3,
    congestion_factor: float = 10.0,
):
    """
    Create a copy of G with hidden congestion.
    Ensures at least one strategic path is sabotaged.
    """
    G_real = G.copy()
    congested_edges = []

    # Random congestion
    for u, v in G_real.edges():
        if random.random() < congestion_prob:
            G_real[u][v]["weight"] *= congestion_factor
            congested_edges.append((u, v))

    # Strategic sabotage
    nodes = list(G.nodes())
    suggested_src, suggested_dst = 0, len(nodes) - 1

    for _ in range(10):
        s, d = random.choice(nodes), random.choice(nodes)
        if s == d:
            continue

        try:
            path = nx.shortest_path(G, s, d, weight="weight")
            if len(path) >= 3:
                u, v = path[1], path[2]
                G_real[u][v]["weight"] *= 20.0
                congested_edges.append((u, v))
                suggested_src, suggested_dst = s, d
                break
        except nx.NetworkXNoPath:
            continue

    return G_real, congested_edges, suggested_src, suggested_dst


# ======================================================
# Session State Initialization
# ======================================================
if "graph_vis" not in st.session_state:
    G_vis = generate_random_network(10, 0.4)
    st.session_state.graph_vis = G_vis
    (
        st.session_state.graph_real,
        st.session_state.congested_edges,
        st.session_state.s_src,
        st.session_state.s_dst,
    ) = apply_congestion(G_vis)
    st.session_state.pos = nx.spring_layout(G_vis, seed=42)

if "agent" not in st.session_state:
    st.session_state.agent = None

if "training_history" not in st.session_state:
    st.session_state.training_history = []


# ======================================================
# Sidebar Configuration
# ======================================================
st.sidebar.title("Configuration")

st.sidebar.subheader("Network Settings")
num_routers = st.sidebar.number_input(
    "Number of Routers", min_value=5, max_value=50, value=10
)
density = st.sidebar.slider("Connection Density", 0.1, 1.0, 0.4)

if st.sidebar.button("New Network"):
    G_vis = generate_random_network(num_routers, density)
    st.session_state.graph_vis = G_vis
    (
        st.session_state.graph_real,
        st.session_state.congested_edges,
        st.session_state.s_src,
        st.session_state.s_dst,
    ) = apply_congestion(G_vis)
    st.session_state.pos = nx.spring_layout(G_vis, seed=42)
    st.session_state.agent = None
    st.session_state.training_history = []
    st.success("New network generated with hidden congestion!")

st.sidebar.divider()

st.sidebar.subheader("Training Params")

# Agent Architecture Selection
agent_architecture = st.sidebar.radio(
    "Agent Architecture",
    options=["Multi-Agent (Distributed)", "Universal (Single Agent)"],
    index=0,
    help="Multi-Agent: Each router has its own agent with memory. Universal: One centralized agent for all routers."
)

# Convert to internal format
agent_type = "multi-agent" if "Multi-Agent" in agent_architecture else "universal"

# Display architecture info
if agent_type == "multi-agent":
    st.sidebar.info("🔹 **Multi-Agent**: Each router learns independently with its own forwarding table")
else:
    st.sidebar.info("🔹 **Universal**: Single agent handles all routing decisions")

episodes = st.sidebar.number_input("Episodes", min_value=10, value=1000)
save_dir = st.sidebar.text_input("Save Directory", value="models_demo")
seed = st.sidebar.number_input("Random Seed", value=0)

# Advanced Training Settings
with st.sidebar.expander("⚙️ Advanced Training Settings"):
    batch_size = st.number_input(
        "Batch Size", 
        min_value=8, 
        max_value=256, 
        value=32,
        help="Number of experiences sampled from replay buffer for each training step"
    )
    
    train_freq = st.number_input(
        "Training Steps per Action",
        min_value=1,
        max_value=10,
        value=1,
        help="How many times to train on batches after each environment step"
    )
    
    buffer_size = st.number_input(
        "Replay Buffer Size",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Maximum number of experiences to store in replay buffer"
    )

st.sidebar.divider()

st.sidebar.subheader("Simulation Params")
src_node = st.sidebar.number_input(
    "Source Node",
    min_value=0,
    max_value=num_routers - 1,
    value=st.session_state.get("s_src", 0),
)
dst_node = st.sidebar.number_input(
    "Dest Node",
    min_value=0,
    max_value=num_routers - 1,
    value=st.session_state.get("s_dst", num_routers - 1),
)
sim_speed = st.sidebar.slider("Simulation Speed (s)", 0.1, 2.0, 0.5)


# ======================================================
# Main UI
# ======================================================
st.title("Packet Routing with Reinforcement Learning")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Topology (Visible Map)")
    fig = draw_network(
        st.session_state.graph_vis,
        st.session_state.pos,
        title="Network Map (Dijkstra sees this)",
    )
    st.pyplot(fig)

with col2:
    G = st.session_state.graph_vis
    st.subheader("Network Stats")
    st.write(f"*Nodes:* {G.number_of_nodes()}")
    st.write(f"*Edges:* {G.number_of_edges()}")
    st.write(f"*Avg Degree:* {(2 * G.number_of_edges() / G.number_of_nodes()):.2f}")
    st.info(
        f"*Hidden Congestion:* {len(st.session_state.congested_edges)} edges have traffic!"
    )

    if st.session_state.agent:
        st.success("Agent Loaded / Trained")
    else:
        st.warning("No Agent Available")


# ======================================================
# Tabs
# ======================================================
tab_train, tab_sim = st.tabs(["Training", "Simulation & Comparison"])

with tab_train:
    st.header("Train the RL Agent")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        if st.button("Start Training"):
            G_real = st.session_state.graph_real
            
            # Re-init agent based on selected architecture
            env_temp = NetworkRoutingEnv(G_real, reward_mode='C')
            state_dim = env_temp._get_state().shape[0]
            action_dim = env_temp.num_nodes
            
            # Import both agent types
            from src.agent import DQNAgent, AgentManager
            
            # Create agent based on user selection
            if agent_type == "multi-agent":
                st.info("🤖 Initializing Multi-Agent Architecture (one agent per router)...")
                agent = AgentManager(
                    state_dim, action_dim, graph=G_real, seed=seed,
                    batch_size=batch_size, buffer_size=buffer_size
                )
                is_multi_agent = True
            else:
                st.info("🤖 Initializing Universal Agent Architecture (single centralized agent)...")
                agent = DQNAgent(
                    state_dim, action_dim, graph=G_real, seed=seed,
                    batch_size=batch_size, buffer_size=buffer_size
                )
                is_multi_agent = False
            
            st.session_state.agent = agent
            st.session_state.agent_type = agent_type  # Store for later use
            st.session_state.training_history = []
            
            # Display training configuration
            st.info(f"📊 Training Config: Batch Size={batch_size}, Train Freq={train_freq}x, Buffer={buffer_size}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            env = NetworkRoutingEnv(G_real, reward_mode='C', seed=seed)
            total_rewards = []
            
            for ep in range(episodes):
                state = env.reset()
                ep_reward = 0
                done = False
                
                while not done:
                    current_node = env.current
                    previous_node = env.previous  # Get previous node to avoid
                    valid_actions = list(G_real.neighbors(current_node))
                    
                    # Action selection with loop prevention (different for each architecture)
                    if is_multi_agent:
                        action = agent.act(state, valid_actions, current_node, avoid_node=previous_node)
                    else:
                        action = agent.act(state, valid_actions, avoid_node=previous_node)
                    
                    next_state, reward, done, info = env.step(action)
                    
                    # Store experience (different for each architecture)
                    if is_multi_agent:
                        agent.remember(state, action, reward, next_state, done, current_node)
                        # Train multiple times per step for better learning
                        for _ in range(train_freq):
                            agent.replay_train(current_node)
                    else:
                        agent.remember(state, action, reward, next_state, done)
                        # Train multiple times per step for better learning
                        for _ in range(train_freq):
                            agent.replay_train()
                    
                    state = next_state
                    ep_reward += reward
                
                total_rewards.append(ep_reward)
                st.session_state.training_history = total_rewards
                
                # Update UI
                if (ep + 1) % 10 == 0:
                    progress = (ep + 1) / episodes
                    progress_bar.progress(progress)
                    
                    avg_reward = np.mean(total_rewards[-50:])
                    eps_display = f"AvgEps: {agent.epsilon:.3f}" if is_multi_agent else f"Eps: {agent.epsilon:.3f}"
                    status_text.text(f"Episode {ep+1}/{episodes} | Avg Reward (last 50): {avg_reward:.2f} | {eps_display}")
                    
                    # Live chart
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(total_rewards, label='Reward')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Total Reward')
                    ax.set_title(f'Training Progress ({agent_type})')
                    ax.legend()
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
            
            st.success(f"✅ Training Complete using {agent_architecture}!")
            
            # Save model
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                if is_multi_agent:
                    agent.save(save_dir)  # AgentManager.save takes directory
                    st.info(f"💾 Multi-agent models saved to {save_dir}/ (one file per router)")
                else:
                    model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
                    agent.save(model_path)
                    st.info(f"💾 Universal agent model saved to {model_path}")

with tab_sim:
    st.header("Simulation & Comparison")
    
    if st.button("Simulate Routing"):
        if st.session_state.agent is None:
            st.error("Please train the agent first!")
        else:
            agent = st.session_state.agent
            G_real = st.session_state.graph_real
            s = src_node
            d = dst_node
            
            # Detect agent type
            is_multi_agent = hasattr(agent, 'agents')
            agent_type_display = "Multi-Agent" if is_multi_agent else "Universal"
            
            st.info(f"🚀 Running simulation with {agent_type_display} architecture")
            
            # -- RL Agent Run --
            # We need to temporarily silence epsilon for evaluation
            old_eps = agent.epsilon
            agent.epsilon = 0.0 # Deterministic
            
            env = NetworkRoutingEnv(G_real, reward_mode='C')
            env.reset(src=s, dst=d)
            
            path_agent = [s]
            cost_agent = 0
            
            placeholder_map = st.empty()
            status_sim = st.empty()
            
            for step in range(num_routers * 2):
                current_node = env.current
                
                # Visualize current step
                fig_sim = draw_network(
                    st.session_state.graph_vis,
                    st.session_state.pos,
                    path=path_agent,
                    path_color="orange",
                    title=f"Step {step}: At node {current_node} ({agent_type_display})"
                )
                placeholder_map.pyplot(fig_sim)
                plt.close(fig_sim)
                time.sleep(sim_speed)
                
                valid_actions = list(G_real.neighbors(current_node))
                if not valid_actions:
                    status_sim.error("Dead end!")
                    break
                
                state = env._get_state()
                previous_node = env.previous  # Get previous node to avoid
                
                # Action selection with loop prevention based on agent type
                if is_multi_agent:
                    action = agent.act(state, valid_actions, current_node, avoid_node=previous_node)
                else:
                    action = agent.act(state, valid_actions, avoid_node=previous_node)
                
                _, reward, done, info = env.step(action)
                
                # Show loop warnings
                if 'immediate_loop' in info:
                    status_sim.warning("⚠️ Immediate backtracking detected!")
                elif 'revisit' in info:
                    status_sim.info(f"ℹ️ Revisiting node (visit #{info.get('visit_count', 1)})")
                
                # Calculate real cost (weight)
                edge_weight = G_real[current_node][action]['weight']
                cost_agent += edge_weight
                
                path_agent.append(action)
                
                if done:
                    if env.current == d:
                        # Show efficiency metrics
                        efficiency = info.get('efficiency', 0)
                        status_sim.success(f"✅ Reached Destination! Cost: {cost_agent:.1f}, Efficiency: {efficiency:.1%}")
                    else:
                        status_sim.error("❌ Max steps reached or excessive looping detected")
                    break
            
            agent.epsilon = old_eps # Restore epsilon
            
            # -- Dijkstra Comparison --
            st.subheader("Comparison with Dijkstra")
            try:
                path_dij = nx.shortest_path(G_real, source=s, target=d, weight='weight')
                cost_dij = nx.shortest_path_length(G_real, source=s, target=d, weight='weight')
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown("**RL Agent**")
                    st.write(f"Path: {path_agent}")
                    st.write(f"Cost: {cost_agent}")
                    
                with col_res2:
                    st.markdown("**Dijkstra (Optimal)**")
                    st.write(f"Path: {path_dij}")
                    st.write(f"Cost: {cost_dij}")
                
                # Final visualization comp
                fig_comp = draw_network(
                    st.session_state.graph_vis,
                    st.session_state.pos,
                    path=path_agent,
                    path_color="orange",
                    title="Agent Path (Orange) vs Dijkstra (Green overlay)"
                )
                
                # Overlay Dijkstra
                # We can't easily overlay efficiently with this simple helper, 
                # but we can just show the agent path and let user compare text.
                # Or we can do a quick hack to show Dijkstra in green if we wanted.
                st.pyplot(fig_comp)
                
            except nx.NetworkXNoPath:
                st.error("No path exists between these nodes!")
