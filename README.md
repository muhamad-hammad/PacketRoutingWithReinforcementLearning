# Packet Routing with Reinforcement Learning

This repository contains a TensorFlow-based DQN demo for learning packet routing policies on a graph topology using NetworkX.

Files added in this commit:
- `requirements.txt` - project dependencies
- `src/env.py` - `NetworkRoutingEnv` (graph environment)
- `src/agent.py` - `DQNAgent` and replay buffer
- `train_dqn_tf.py` - training script for the DQN (saves model and plots)
- `evaluate.py` - evaluation and comparison with Dijkstra
- `run_demo.py` - helper to run a short demo training + evaluation
- `tests/test_env.py` - minimal unit test for the environment

Quick start (Windows PowerShell):

```powershell
# create and activate venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run a short demo training
python run_demo.py
```

Notes:
- This demo uses CPU TensorFlow by default. No GPU-specific code is required.
- The training is intentionally small in the demo; tune hyperparameters in `train_dqn_tf.py` for larger experiments.
