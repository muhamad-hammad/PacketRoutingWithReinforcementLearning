import tensorflow as tf
try:
    print("TensorFlow version:", tf.__version__)
    x = tf.random.normal([100, 100])
    y = tf.reduce_sum(x)
    print("Computation result:", y.numpy())
    print("TensorFlow is working.")
except Exception as e:
    print("TensorFlow failed:", e)
