
import tensorflow as tf
import os

# Force CPU usage to avoid potential GPU memory issues if any (though user likely on CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def debug_log(msg):
    print(f"[DEBUG] {msg}", flush=True)
