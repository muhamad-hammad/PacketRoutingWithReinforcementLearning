"""Tkinter GUI for PacketRoutingWithReinforcementLearning

A visual interface for training and evaluating deep Q-learning agents on network routing tasks.

This is a cleaned, single-copy version of the GUI module. It embeds a Matplotlib
canvas inside Tkinter and exposes controls to train/load/evaluate a DQN agent
and to run step-by-step simulations.
"""

import os
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Project imports
from train_dqn_tf import train, build_sample_graph
from src.agent import DQNAgent
import evaluate
from src.env import NetworkRoutingEnv
import random

LOG_QUEUE = queue.Queue()


def log(msg: str):
    """Queue a log message for the GUI log area."""
    LOG_QUEUE.put(msg)


class App(tk.Tk):
    """Main application window for the packet routing demo."""

    def __init__(self):
        super().__init__()
        self.title("Packet Routing RL - Demo GUI")
        self.geometry("1200x700")  # Made window wider to accommodate new controls

        # Controls frame
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        # Training controls
        ttk.Label(frm, text="Episodes:").grid(row=0, column=0, sticky=tk.W)
        self.episodes_var = tk.IntVar(value=300)
        ttk.Entry(frm, textvariable=self.episodes_var, width=8).grid(row=0, column=1)

        ttk.Label(frm, text="Save dir:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.save_dir_var = tk.StringVar(value='models_demo')
        ttk.Entry(frm, textvariable=self.save_dir_var, width=20).grid(row=0, column=3)

        ttk.Label(frm, text="Seed:").grid(row=0, column=4, sticky=tk.W, padx=(10, 0))
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(frm, textvariable=self.seed_var, width=6).grid(row=0, column=5)

        # Training control frame with buttons and menu
        train_ctrl_frame = ttk.Frame(frm)
        train_ctrl_frame.grid(row=0, column=6, columnspan=3, padx=(10, 0))

        # Training buttons frame for main buttons
        train_btn_frame = ttk.Frame(train_ctrl_frame)
        train_btn_frame.pack(side=tk.TOP, fill=tk.X)

        # Training control menu
        self.train_menu = ttk.Menubutton(train_btn_frame, text="▼")
        self.train_menu.pack(side=tk.RIGHT)
        
        train_menu = tk.Menu(self.train_menu, tearoff=0)
        self.train_menu['menu'] = train_menu
        
        # Training control states and menu items
        self.training_active = False
        self.training_paused = False
        
        train_menu.add_command(label="Stop", command=self.stop_training)
        train_menu.add_command(label="Pause", command=self.pause_training)
        train_menu.add_command(label="Resume", command=self.resume_training)
        train_menu.add_separator()
        train_menu.add_command(label="Restart", command=self.restart_training)
        
        # Main training buttons
        self.btn_train = ttk.Button(train_btn_frame, text="Train", command=self.on_train)
        self.btn_train.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_live_train = ttk.Button(train_btn_frame, text="Live Train", command=self.on_live_train)
        self.btn_live_train.pack(side=tk.LEFT, padx=(0, 5))

        # Model management buttons
        btn_load = ttk.Button(train_btn_frame, text="Load Model", command=self.on_load)
        btn_load.pack(side=tk.LEFT, padx=(0, 5))

        btn_eval = ttk.Button(train_btn_frame, text="Evaluate", command=self.on_evaluate)
        btn_eval.pack(side=tk.LEFT, padx=(0, 5))
        
        # Network configuration controls
        ttk.Label(frm, text="Routers:").grid(row=0, column=9, sticky=tk.W, padx=(10, 0))
        self.num_routers_var = tk.IntVar(value=5)
        ttk.Entry(frm, textvariable=self.num_routers_var, width=6).grid(row=0, column=10, sticky=tk.W)
        
        ttk.Label(frm, text="Density:").grid(row=0, column=11, sticky=tk.W, padx=(10, 0))
        self.density_var = tk.DoubleVar(value=0.4)
        ttk.Entry(frm, textvariable=self.density_var, width=6).grid(row=0, column=12, sticky=tk.W)
        
        btn_new_network = ttk.Button(frm, text="New Network", command=self.on_new_network)
        btn_new_network.grid(row=0, column=13, padx=(10, 0))

        # Simulation controls
        ttk.Label(frm, text="Src:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.src_var = tk.IntVar(value=0)
        ttk.Entry(frm, textvariable=self.src_var, width=6).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(frm, text="Dst:").grid(row=1, column=2, sticky=tk.W, padx=(10, 0))
        self.dst_var = tk.IntVar(value=4)
        ttk.Entry(frm, textvariable=self.dst_var, width=6).grid(row=1, column=3, sticky=tk.W)

        ttk.Label(frm, text="Delay(s):").grid(row=1, column=4, sticky=tk.W, padx=(10, 0))
        self.delay_var = tk.DoubleVar(value=1.0)
        ttk.Entry(frm, textvariable=self.delay_var, width=6).grid(row=1, column=5, sticky=tk.W)

        btn_sim = ttk.Button(frm, text="Simulate", command=self.on_simulate)
        btn_sim.grid(row=1, column=6, padx=(10, 0))
        
        btn_compare = ttk.Button(frm, text="Compare with Dijkstra", command=self.on_compare)
        btn_compare.grid(row=1, column=7, padx=(5, 0))

        # Live training controls (slow visualization)
        ttk.Label(frm, text="Step delay(s):").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        self.step_delay_var = tk.DoubleVar(value=0.5)
        ttk.Entry(frm, textvariable=self.step_delay_var, width=6).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(frm, text="Ep delay(s):").grid(row=2, column=2, sticky=tk.W, padx=(10, 0))
        self.ep_delay_var = tk.DoubleVar(value=0.3)
        ttk.Entry(frm, textvariable=self.ep_delay_var, width=6).grid(row=2, column=3, sticky=tk.W)

        # Main content area: left=plot, right=log
        content = ttk.Frame(self)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Plot frame (left)
        plot_frame = ttk.LabelFrame(content, text="Simulation")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log area (right)
        log_frame = ttk.LabelFrame(content, text="Log")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, ipadx=8)

        self.log_text = tk.Text(log_frame, wrap=tk.NONE, width=50)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scroll_y.set

        # Internal state
        self.agent = None
        self.G = build_sample_graph()
        # layout positions for plotting
        self.pos = nx.spring_layout(self.G, seed=42)
        
        # Training control state
        self.training_thread = None
        self.training_stop = threading.Event()
        self.training_pause = threading.Event()

        # Periodic log updater
        self.after(200, self._drain_log_queue)

    def _drain_log_queue(self):
        try:
            while True:
                msg = LOG_QUEUE.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(200, self._drain_log_queue)
        
    def _generate_random_network(self, num_nodes, density):
        """Generate a random network with weighted edges.
        
        Args:
            num_nodes: Number of routers/nodes in the network
            density: Probability of edge creation between nodes (0-1)
        """
        # Create empty graph
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(num_nodes))
        
        # Add random edges with weights
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < density:
                    # Random weight between 1 and 10
                    weight = random.randint(1, 10)
                    G.add_edge(i, j, weight=weight)
        
        # Ensure graph is connected - add minimum edges if needed
        components = list(nx.connected_components(G))
        while len(components) > 1:
            # Connect two random nodes from different components
            comp1 = random.choice(list(components[0]))
            comp2 = random.choice(list(components[1]))
            weight = random.randint(1, 10)
            G.add_edge(comp1, comp2, weight=weight)
            components = list(nx.connected_components(G))
        
        return G
        
    def on_new_network(self):
        """Create a new network with specified number of routers."""
        try:
            num_routers = int(self.num_routers_var.get())
            density = float(self.density_var.get())
            
            if num_routers < 2:
                messagebox.showerror("Invalid input", "Number of routers must be at least 2")
                return
                
            if density <= 0 or density > 1:
                messagebox.showerror("Invalid input", "Density must be between 0 and 1")
                return
            
            # Generate new network
            self.G = self._generate_random_network(num_routers, density)
            
            # Reset agent since network changed
            self.agent = None
            
            # Calculate new layout
            self.pos = nx.spring_layout(self.G, seed=42)
            
            # Update plot
            self.ax.clear()
            nx.draw_networkx_nodes(self.G, self.pos, node_color='lightblue', node_size=500, ax=self.ax)
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
            nx.draw_networkx_edges(self.G, self.pos, alpha=0.5, ax=self.ax)
            edge_labels = nx.get_edge_attributes(self.G, 'weight')
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, font_size=8, ax=self.ax)
            
            self.ax.set_title(f"New Network: {num_routers} routers, {self.G.number_of_edges()} links")
            self.ax.axis('off')
            self.canvas.draw()
            
            # Log info
            log(f"Created new network with {num_routers} routers and {self.G.number_of_edges()} links")
            log(f"Average node degree: {2*self.G.number_of_edges()/self.G.number_of_nodes():.1f}")
            log("Note: Previous model cleared since network topology changed")
            
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))

    def _run_in_thread(self, fn, *args, **kwargs):
        def _wrap():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                log(f"ERROR: {e}")
                tb = traceback.format_exc()
                log(tb)
        t = threading.Thread(target=_wrap, daemon=True)
        t.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.training_stop.set()
            self.training_pause.clear()
            log("Stopping training...")
            self.training_thread.join()
            log("Training stopped.")
            self.training_active = False
            self.current_episode = 0

    def pause_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.training_pause.set()
            log("Training paused...")
            self.training_active = False

    def resume_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.training_pause.clear()
            log("Resuming training...")
            self.training_active = True

    def restart_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training()
        self.on_train()  # Start fresh training

    def update_training_buttons(self):
        is_active = self.training_thread and self.training_thread.is_alive()
        is_paused = is_active and self.training_pause.is_set()
        
        # Update button states
        self.btn_train['state'] = 'disabled' if is_active else 'normal'
        self.btn_live_train['state'] = 'disabled' if is_active else 'normal'
        
        # Update menu labels
        menu = self.train_menu['menu']
        menu.entryconfigure("Stop", state='normal' if is_active else 'disabled')
        menu.entryconfigure("Pause", state='normal' if is_active and not is_paused else 'disabled')
        menu.entryconfigure("Resume", state='normal' if is_paused else 'disabled')
        menu.entryconfigure("Restart", state='normal' if is_active else 'disabled')

    def on_train(self):
        episodes = int(self.episodes_var.get())
        save_dir = self.save_dir_var.get()
        seed = int(self.seed_var.get())

        os.makedirs(save_dir, exist_ok=True)
        log(f"Starting training: episodes={episodes}, save_dir={save_dir}, seed={seed}")

        # Reset control flags
        self.training_stop.clear()
        self.training_pause.clear()
        self.current_episode = 0
        self.total_episodes = episodes
        self.training_active = True

        def _train_task():
            try:
                # Use the current GUI graph
                G = self.G
                env = NetworkRoutingEnv(G, reward_mode='C', seed=seed)
                state_dim = env._get_state().shape[0]
                action_dim = env.num_nodes
                agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed)

                rewards = []

                for ep in range(episodes):
                    if self.training_stop.is_set():
                        log("Training stopped.")
                        break

                    while self.training_pause.is_set():
                        time.sleep(0.1)
                        if self.training_stop.is_set():
                            break

                    self.current_episode = ep + 1
                    state = env.reset()
                    total_reward = 0.0
                    done = False

                    while not done:
                        if self.training_stop.is_set():
                            break

                        while self.training_pause.is_set():
                            time.sleep(0.1)
                            if self.training_stop.is_set():
                                break

                        valid_actions = list(G.neighbors(env.current))
                        action = agent.act(state, valid_actions)
                        next_state, reward, done, info = env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay_train()
                        state = next_state
                        total_reward += reward

                    if not self.training_stop.is_set():
                        rewards.append(total_reward)
                        avg50 = sum(rewards[-50:]) / min(len(rewards), 50)
                        log(f'Ep {ep+1}/{episodes}  TotalReward={total_reward:.2f}  Avg50={avg50:.2f}  Eps={agent.epsilon:.3f}')

                if not self.training_stop.is_set():
                    # Save model
                    model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
                    agent.save(model_path)
                    log(f'Training completed. Model saved to {model_path}')

                self.agent = agent

            except Exception as e:
                log(f"Training error: {e}")
                log(traceback.format_exc())
            finally:
                self.training_active = False
                self.after(100, self.update_training_buttons)

        self.training_thread = threading.Thread(target=_train_task, daemon=True)
        self.training_thread.start()
        self.update_training_buttons()

    def on_load(self):
        path = filedialog.askopenfilename(title="Select .keras model file",
                                          filetypes=[("Keras model", "*.keras"), ("All files", "*")])
        if not path:
            return
        log(f"Loading model from {path}")

        try:
            state_dim = NetworkRoutingEnv(self.G)._get_state().shape[0]
            self.agent = DQNAgent(state_dim, self.G.number_of_nodes(), graph=self.G)
            self.agent.load(path)
            log("Model loaded successfully.")
        except Exception as e:
            log(f"Failed to load model: {e}")
            tb = traceback.format_exc()
            log(tb)

    def on_evaluate(self):
        if self.agent is None:
            messagebox.showinfo("No model", "Please train or load a model first.")
            return
        log("Starting evaluation (this runs in background)...")

        def _eval_task():
            stats = evaluate.compare_with_dijkstra(self.agent, self.G, trials=50)
            evaluate.analyze_stats(stats)
            log("Evaluation complete. See console output and models_demo/*.png for route visualizations.")

        self._run_in_thread(_eval_task)

    def on_simulate(self):
        src = int(self.src_var.get())
        dst = int(self.dst_var.get())
        delay = float(self.delay_var.get())

        if src not in self.G.nodes() or dst not in self.G.nodes():
            messagebox.showerror("Invalid nodes", "Source or destination node not in graph.")
            return

        log(f"Starting simulation: {src} -> {dst} (delay={delay}s)")

        def _sim_task():
            try:
                self._run_simulation(src, dst, delay)
            except Exception as e:
                log(f"Simulation error: {e}")
                log(traceback.format_exc())

        self._run_in_thread(_sim_task)

    def on_live_train(self):
        episodes = int(self.episodes_var.get())
        save_dir = self.save_dir_var.get()
        seed = int(self.seed_var.get())
        step_delay = float(self.step_delay_var.get())
        ep_delay = float(self.ep_delay_var.get())

        os.makedirs(save_dir, exist_ok=True)
        log(f"Starting LIVE training: episodes={episodes}, step_delay={step_delay}, ep_delay={ep_delay}, save_dir={save_dir}")

        # Reset control flags
        self.training_stop.clear()
        self.training_pause.clear()
        self.current_episode = 0
        self.total_episodes = episodes
        self.training_active = True

        def _run():
            try:
                # Use the current GUI graph instead of creating a new one
                G = self.G
                # Update position layout for consistency
                self.pos = nx.spring_layout(G, seed=42)
                
                env = NetworkRoutingEnv(G, reward_mode='C', seed=seed)
                state_dim = env._get_state().shape[0]
                action_dim = env.num_nodes
                agent = DQNAgent(state_dim, action_dim, graph=G, seed=seed)

                rewards = []

                for ep in range(episodes):
                    if self.training_stop.is_set():
                        log("Live training stopped.")
                        break

                    while self.training_pause.is_set():
                        time.sleep(0.1)
                        if self.training_stop.is_set():
                            break
                    state = env.reset()
                    total_reward = 0.0
                    done = False
                    traversed = []
                    visited = [env.current]
                    current = env.current

                    # show start of episode
                    self._schedule_plot_update(current, env.dst, visited[:], traversed[:], ep+1, total_reward, agent.epsilon)

                    while not done:
                        valid_actions = list(G.neighbors(env.current))
                        action = agent.act(state, valid_actions)
                        next_state, reward, done, info = env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay_train()
                        state = next_state
                        total_reward += reward

                        traversed.append((current, action))
                        visited.append(action)
                        current = action

                        # update plot on main thread
                        self._schedule_plot_update(current, env.dst, visited[:], traversed[:], ep+1, total_reward, agent.epsilon)

                        time.sleep(step_delay)

                    rewards.append(total_reward)
                    avg50 = sum(rewards[-50:]) / min(len(rewards), 50)
                    log(f'Ep {ep+1}/{episodes}  TotalReward={total_reward:.2f}  Avg50={avg50:.2f}  Eps={agent.epsilon:.3f}')

                    # brief pause between episodes
                    time.sleep(ep_delay)

                # save model
                model_path = os.path.join(save_dir, 'dqn_routing_tf.keras')
                agent.save(model_path)
                log(f'LIVE training finished. Model saved to {model_path}')

            except Exception as e:
                log(f"Live training error: {e}")
                log(traceback.format_exc())

        self._run_in_thread(_run)

    def _schedule_plot_update(self, current, dst, visited, traversed, episode, total_reward, eps):
        def _ui_update():
            try:
                self.ax.clear()
                node_colors = []
                for n in self.G.nodes():
                    if n == current:
                        node_colors.append('red')
                    elif n == dst:
                        node_colors.append('green')
                    elif n in visited:
                        node_colors.append('yellow')
                    else:
                        node_colors.append('lightblue')

                nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=500, ax=self.ax)
                nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
                nx.draw_networkx_edges(self.G, self.pos, alpha=0.3, ax=self.ax)
                
                # Draw edge weights
                edge_labels = nx.get_edge_attributes(self.G, 'weight')
                nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, font_size=8, ax=self.ax)
                
                if traversed:
                    nx.draw_networkx_edges(self.G, self.pos, edgelist=traversed, edge_color='r', width=2.5, ax=self.ax)

                self.ax.set_title(f"LIVE Train Ep {episode}  Reward={total_reward:.2f}  Eps={eps:.3f}")
                self.ax.axis('off')
                self.canvas.draw()

            except Exception as e:
                log(f"Error updating LIVE UI: {e}")
                log(traceback.format_exc())

        self.after(1, _ui_update)

    def on_compare(self):
        if self.agent is None:
            messagebox.showinfo("No model", "Please train or load a model first.")
            return
            
        src = int(self.src_var.get())
        dst = int(self.dst_var.get())
        delay = float(self.delay_var.get())

        if src not in self.G.nodes() or dst not in self.G.nodes():
            messagebox.showerror("Invalid nodes", "Source or destination node not in graph.")
            return

        log(f"Starting comparison: {src} -> {dst} (delay={delay}s)")
        
        def _compare_task():
            try:
                # Save original epsilon and set to 0 for deterministic behavior
                original_epsilon = self.agent.epsilon
                self.agent.epsilon = 0.0
                
                # Get Dijkstra's path
                try:
                    dijkstra_path = nx.shortest_path(self.G, src, dst, weight='weight')
                    dijkstra_length = nx.shortest_path_length(self.G, src, dst, weight='weight')
                except nx.NetworkXNoPath:
                    log("No path exists between source and destination!")
                    self.agent.epsilon = original_epsilon
                    return
                
                # Get RL agent's path
                env = NetworkRoutingEnv(self.G, reward_mode='C')
                env.reset(src=src, dst=dst)
                current = src
                rl_path = [src]
                rl_length = 0
                step = 0
                
                while step < env.max_steps:
                    state = env._get_state()
                    neighbors = list(self.G.neighbors(current))
                    action = self.agent.act(state, neighbors)
                    _, reward, done, _ = env.step(action)
                    rl_path.append(action)
                    if len(rl_path) > 1:
                        rl_length += self.G[rl_path[-2]][rl_path[-1]]['weight']
                    current = action
                    step += 1
                    if done:
                        break
                
                # Create path edges for visualization
                dijkstra_edges = list(zip(dijkstra_path[:-1], dijkstra_path[1:]))
                rl_edges = list(zip(rl_path[:-1], rl_path[1:]))
                
                # Clear and setup the plot
                self.ax.clear()
                
                # Draw basic graph structure
                nx.draw_networkx_nodes(self.G, self.pos, node_color='lightblue', node_size=500, ax=self.ax)
                nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
                nx.draw_networkx_edges(self.G, self.pos, alpha=0.2, ax=self.ax)
                edge_labels = nx.get_edge_attributes(self.G, 'weight')
                nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, font_size=8, ax=self.ax)
                
                # Draw Dijkstra's path
                nx.draw_networkx_edges(self.G, self.pos, edgelist=dijkstra_edges, 
                                     edge_color='blue', width=2.5, ax=self.ax, 
                                     label=f"Dijkstra (cost={dijkstra_length:.2f})")
                
                # Draw RL path
                nx.draw_networkx_edges(self.G, self.pos, edgelist=rl_edges, 
                                     edge_color='red', width=2.5, ax=self.ax, 
                                     label=f"RL Agent (cost={rl_length:.2f})")
                
                # Highlight source and destination
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=[src], 
                                     node_color='green', node_size=500, ax=self.ax)
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=[dst], 
                                     node_color='red', node_size=500, ax=self.ax)
                
                self.ax.set_title(f"Path Comparison: {src} -> {dst}")
                self.ax.legend()
                self.ax.axis('off')
                self.canvas.draw()
                
                # Log results
                log(f"Dijkstra's path: {' -> '.join(map(str, dijkstra_path))} (cost={dijkstra_length:.2f})")
                log(f"RL Agent path: {' -> '.join(map(str, rl_path))} (cost={rl_length:.2f})")
                if rl_length == dijkstra_length:
                    log("RL Agent found optimal path! 🎉")
                elif rl_length > dijkstra_length:
                    log(f"RL Agent path is {(rl_length/dijkstra_length - 1)*100:.1f}% longer than optimal")
                
                # Restore original epsilon
                self.agent.epsilon = original_epsilon
                
            except Exception as e:
                log(f"Comparison error: {e}")
                log(traceback.format_exc())
                # Restore epsilon even on error
                if self.agent:
                    self.agent.epsilon = original_epsilon
                
        self._run_in_thread(_compare_task)

    def _run_simulation(self, src, dst, delay=1.0):
        # Save original epsilon and set to 0 for deterministic behavior
        original_epsilon = self.agent.epsilon if self.agent else 0.0
        if self.agent:
            self.agent.epsilon = 0.0
        
        env = NetworkRoutingEnv(self.G, reward_mode='C')
        env.reset(src=src, dst=dst)

        visited = [src]
        traversed = []
        current = src
        step = 0
        max_steps = env.max_steps

        while step < max_steps:
            self.ax.clear()
            node_colors = []
            for n in self.G.nodes():
                if n == current:
                    node_colors.append('red')
                elif n == dst:
                    node_colors.append('green')
                elif n in visited:
                    node_colors.append('yellow')
                else:
                    node_colors.append('lightblue')

            nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=500, ax=self.ax)
            nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
            nx.draw_networkx_edges(self.G, self.pos, alpha=0.5, ax=self.ax)
            edge_labels = nx.get_edge_attributes(self.G, 'weight')
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, font_size=8, ax=self.ax)

            if traversed:
                nx.draw_networkx_edges(self.G, self.pos, edgelist=traversed, edge_color='r', width=2.5, ax=self.ax)

            self.ax.set_title(f"Simulation: {src} -> {dst}  step={step}")
            self.ax.axis('off')
            self.canvas.draw()

            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                log("No neighbors - stopping simulation")
                break

            state = env._get_state()
            try:
                action = self.agent.act(state, neighbors)
                used_agent = True
            except Exception:
                try:
                    path = nx.shortest_path(self.G, current, dst, weight='weight')
                    if len(path) >= 2:
                        action = path[1]
                    else:
                        break
                except nx.NetworkXNoPath:
                    action = neighbors[0]
                used_agent = False

            _, reward, done, _ = env.step(action)
            traversed.append((current, action))
            visited.append(action)

            self.ax.annotate('', xy=self.pos[action], xytext=self.pos[current], arrowprops=dict(arrowstyle='->', color='magenta', lw=2))
            self.canvas.draw()

            log(f"Step {step}: {current} -> {action}  {'agent' if used_agent else 'dijkstra'}")

            time.sleep(delay)
            current = action
            step += 1

            if done:
                log(f"Reached destination {dst} in {step} steps")
                break
        
        # Restore original epsilon
        if self.agent:
            self.agent.epsilon = original_epsilon


if __name__ == '__main__':
    app = App()
    app.mainloop()