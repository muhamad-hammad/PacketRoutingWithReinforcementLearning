import os
import time
import random
import numpy as np
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf

from src.env import NetworkRoutingEnv
from src.agent import DQNAgent
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
episodes = st.sidebar.number_input("Episodes", min_value=10, value=1000)
save_dir = st.sidebar.text_input("Save Directory", value="models_demo")
seed = st.sidebar.number_input("Random Seed", value=0)

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