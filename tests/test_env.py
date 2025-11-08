import pytest
import networkx as nx
from src.env import NetworkRoutingEnv


def test_env_step_and_reset():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0)])
    env = NetworkRoutingEnv(G, reward_mode='C', seed=0)
    s = env.reset()
    assert len(s) == env.num_nodes * 3 + 1
    neighbors = list(G.neighbors(env.current))
    assert neighbors, 'start node should have neighbors'
    ns, r, done, info = env.step(neighbors[0])
    assert isinstance(r, float)
    assert isinstance(done, bool)
