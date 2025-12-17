import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def debug_log(msg):
    print(f"[DEBUG] {msg}", flush=True)
