"""Simple Tkinter GUI for PacketRoutingWithReinforcementLearning

Features:
- Train the DQN agent (runs in background thread)
- Load a saved model
- Run evaluation (compare with Dijkstra)
- Show log output in the GUI

This is intentionally minimal and dependency-free (uses built-in Tkinter).
"""
import threading
import queue
import os
import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Import project modules
from train_dqn_tf import train, build_sample_graph
from src.agent import DQNAgent
import evaluate
from src.env import NetworkRoutingEnv

LOG_QUEUE = queue.Queue()


def log(msg):
    LOG_QUEUE.put(msg)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Packet Routing RL - Demo GUI")
        self.geometry("800x600")

        # Controls frame
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(frm, text="Episodes:").grid(row=0, column=0, sticky=tk.W)
        self.episodes_var = tk.IntVar(value=300)
        ttk.Entry(frm, textvariable=self.episodes_var, width=8).grid(row=0, column=1)

        ttk.Label(frm, text="Save dir:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.save_dir_var = tk.StringVar(value='models_demo')
        ttk.Entry(frm, textvariable=self.save_dir_var, width=20).grid(row=0, column=3)

        ttk.Label(frm, text="Seed:").grid(row=0, column=4, sticky=tk.W, padx=(10, 0))
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(frm, textvariable=self.seed_var, width=6).grid(row=0, column=5)

        # Buttons
        btn_train = ttk.Button(frm, text="Train", command=self.on_train)
        btn_train.grid(row=0, column=6, padx=(10, 0))

        btn_load = ttk.Button(frm, text="Load Model", command=self.on_load)
        btn_load.grid(row=0, column=7, padx=(5, 0))

        btn_eval = ttk.Button(frm, text="Evaluate", command=self.on_evaluate)
        btn_eval.grid(row=0, column=8, padx=(5, 0))

        # Log area
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.log_text = tk.Text(log_frame, wrap=tk.NONE)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = scroll_y.set

        # Internal state
        self.agent = None
        self.G = build_sample_graph()

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

    def on_train(self):
        episodes = int(self.episodes_var.get())
        save_dir = self.save_dir_var.get()
        seed = int(self.seed_var.get())

        os.makedirs(save_dir, exist_ok=True)
        log(f"Starting training: episodes={episodes}, save_dir={save_dir}, seed={seed}")

        def _train_task():
            agent, G = train(episodes=episodes, save_dir=save_dir, seed=seed)
            self.agent = agent
            self.G = G
            log("Training finished and agent assigned.")

        self._run_in_thread(_train_task)

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


if __name__ == '__main__':
    app = App()
    app.mainloop()
