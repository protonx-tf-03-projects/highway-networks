# TODO 2:  

from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.ops.gen_array_ops import shape

class HighwayMLP(tf.keras.Model):
  """
  Highway MLP Layer
  """
  def __init__(self, input_size, t_bias=-2, acti_h = tf.nn.relu, acti_t = tf.nn.tanh):
    super(HighwayMLP, self).__init__()
    self.acti_h = acti_h
    self.acti_t = acti_t
    self.t_bias = t_bias

    # TODO 
    self.W = tf.Variable(tf.truncated_normal([input_size, input_size], stddev=0.1), name="weight")
    self.b = tf.Variable(tf.constant(self.t_bias, shape=[input_size]), name="bias")
    self.x = None

    # Dense H
    self.h = acti_h(tf.matmul(self.x, self.W) + self.b, name="H_gate")
    # Dense T

    self.t = acti_t(tf.matmul(self.x, self.W) + self.b, name="transform_gate")



    pass

  def call(self, x):
    # Do Highway: y = H(x,WH)· T(x,WT) + x · C(x,WC).
    self.x = x
    y = tf.add(tf.multiply(self.h, self.t), tf.multiply(x, tf.sub(1.0, self.t)), name="highway_layer")


class HighwayNetwork(tf.keras.Model): 
  """
  Highway Network with several layers
  """
  def __init__(self, input_size, output_size):
    super(HighwayNetwork, self).__init__()
    self.mlplayers = [
      # TO DO
      tf.keras.Input(shape=input_size),
      tf.layers.Dense(71),
      HighwayMLP.call(),
      tf.layers.Dense(output_size, activations=tf.nn.softmax)
    ]

    # Classification layer

  def call(self, x):
    # Run input on these mlp layers
    for layer in self.mlplayers:
      if layer == 0:
        prev_y = layer(self.input_size)
      elif layer == self.mlplayers - 1:
        y = layer(self.output_size)
      else:
        prev_y = layer(self.prev_y)

    # pass output to classification layer



model = HighwayNetwork(784, 10)
