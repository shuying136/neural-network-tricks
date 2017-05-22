def highway(x, size, activation, carry_bias=-1.0):
  W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
  b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

  W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
  b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

  T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
  H = activation(tf.matmul(x, W) + b, name="activation")
  C = tf.sub(1.0, T, name="carry_gate")

  y = tf.add(tf.mul(H, T), tf.mul(x, C), "y")
  return y
