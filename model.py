# TODO 2:
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
import numpy as np


class Dense(layers.Layer):
  def __init__(self, units, bias=0.1):
    super(Dense, self).__init__()
    self.units = units
    self.bias = bias

  def build(self, input_shape):
    self.w = self.add_weight(
      name="w",
      shape=(input_shape[-1], self.units),
      initializer="random_normal",
      trainable=True,
    )

    self.b = tf.Variable(tf.constant(self.bias, shape=self.units), name='bias', trainable=True)


  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

class HighwayMLP(layers.Layer):
  """
  Highway MLP Layer
  """
  def __init__(self, t_bias=-2.0, acti_h = tf.nn.relu, acti_t = tf.nn.tanh):
    super(HighwayMLP, self).__init__()
    self.acti_h = acti_h
    self.acti_t = acti_t
    self.bias_init = t_bias
    self.t = Dense(50, self.bias_init)
    self.h = Dense(50)
    self.projection = Dense(50)

  def call(self, x, training = False):
    # Do Highway: y = H(x,WH)· T(x,WT) + x · C(x,WC).

    # TODO
    # Dense H
    dense_h = self.h(x, training = training)
    dense_h = self.acti_h(dense_h)

    # Dense T
    dense_t = self.t(x,training = training)
    dense_t = self.acti_t(dense_t)

    dense_x = self.projection(x)


    y = tf.add(tf.multiply(dense_h,dense_t) , tf.multiply(dense_x, (1 - dense_t)))

    return y


class HighwayNetwork(tf.keras.Model):
  """
  Highway Network with several layers
  """
  def __init__(self, output_size = 10, num_of_layers = 5):
    super(HighwayNetwork, self).__init__()
    self.output_size = output_size
    self.mlplayers = [
      HighwayMLP() for _ in range(num_of_layers)
    ]

    # Classification layer
    self.classificationLayer = tf.keras.layers.Dense(
      self.output_size, activation='softmax'
    )

  def call(self, x):
    # Run input on these mlp layers

    for layer in self.mlplayers:
      x = layer(x)


    # pass output to classification layer
    out = self.classificationLayer(x)

    return out

model = HighwayNetwork()