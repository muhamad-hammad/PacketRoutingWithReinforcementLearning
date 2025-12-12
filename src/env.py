import random
import numpy as np
import networkx as nx


class NetworkRoutingEnv:
    """A small Gym-like routing environment on a NetworkX graph.
    State encoding (default): concatenation of one-hot current node,
    one-hot destination node, one-hot previous node (or zeros if none),
    and a scalar normalized hop count (as a 1-element array).
    """

    def __init__(self, graph: nx.Graph, max_steps=None, reward_mode='C', seed=None):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.max_steps = max_steps or (2 * self.num_nodes)
        self.reward_mode = reward_mode  # 'A' or 'C'
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.reset()

    def reset(self, src=None, dst=None):
        nodes = list(self.graph.nodes())

        if src is None:
            self.src = random.choice(nodes)
        else:
            self.src = src

        if dst is None:
            self.dst = random.choice(nodes)
            while self.dst == self.src:
                self.dst = random.choice(nodes)
        else:
            self.dst = dst

        self.current = self.src
        self.previous = None
        self.steps = 0
        self.visited = {self.current}

        return self._get_state()

    def _one_hot(self, idx):
        v = np.zeros(self.num_nodes, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _get_state(self):
        cur_v = self._one_hot(self.current)
        dst_v = self._one_hot(self.dst)
        prev_v = self._one_hot(self.previous) if self.previous is not None else np.zeros(self.num_nodes, dtype=np.float32)
        hop_norm = np.array([self.steps / float(self.max_steps)], dtype=np.float32)

        return np.concatenate([cur_v, dst_v, prev_v, hop_norm])

    def neighbors(self):
        return list(self.graph.neighbors(self.current))

    def step(self, action):
        """Take action = next_node index. Returns (state, reward, done, info)"""

        self.steps += 1
        info = {}

        if action not in self.graph.neighbors(self.current):
            # Invalid action: heavy penalty and terminate episode
            reward = -100.0
            done = True
            return self._get_state(), reward, done, {'invalid_action': True}

        cost = float(self.graph[self.current][action].get('weight', 1.0))

        # detect revisit
        revisit_penalty = 0.0
        if action in self.visited:
            revisit_penalty = -0.5

        self.previous = self.current
        self.current = action
        self.visited.add(self.current)

        done = (self.current == self.dst) or (self.steps >= self.max_steps)

        # reward modes
        if self.reward_mode == 'A':
            reward = 100.0 if self.current == self.dst else -cost
        else:  # 'C' (recommended): -cost, +100 arrival, small revisit penalty
            reward = 100.0 if self.current == self.dst else -cost
            if not done and revisit_penalty != 0.0:
                reward += revisit_penalty

        return self._get_state(), float(reward), bool(done), info

    def render(self, draw_pos=None, show=False, path=None):
        import matplotlib.pyplot as plt

        pos = draw_pos or nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(6, 4))

        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue')

        if path is not None:
            edges_in_path = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color='orange')
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges_in_path, edge_color='orange', width=2)

        if show:
            plt.show()

        plt.close()


if __name__ == '__main__':
    # tiny smoke test
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 2.5)])

    env = NetworkRoutingEnv(G, reward_mode='C', seed=0)
    s = env.reset()

    print('state length:', len(s))

    ns, r, d, info = env.step(1)
    print('step->', r, d, info)