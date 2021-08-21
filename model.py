# TODO 2:  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations

class HighwayMLP(tf.keras.Model):
  """
  Highway MLP Layer
  """
  def __init__(self, input_size, t_bias=-2.0, acti_h = tf.nn.relu, acti_t = tf.nn.tanh):
    super(HighwayMLP, self).__init__()
    self.acti_h = acti_h
    self.acti_t = acti_t
    self.size = input_size
    self.bias_init = t_bias

    self.W_H = tf.Variable(
      tf.random.truncated_normal((self.size,self.size), stddev=0.1), name='weight'
    )
    self.b_H = tf.Variable(tf.constant(self.bias_init, shape=self.size), name='bias')

    self.W_T = tf.Variable(
      tf.random.truncated_normal((self.size,self.size), stddev=0.1), name='weight'
    )
    self.b_T = tf.Variable(tf.constant(self.bias_init, shape=self.size), name='bias')
  
  def call(self, x):
    # Do Highway: y = H(x,WH)· T(x,WT) + x · C(x,WC).
    # Dense H
    self.h = self.acti_h(tf.matmul(x, self.W_H) + self.b_H)
    # Dense T
    self.t = self.acti_t(tf.matmul(x, self.W_T) + self.b_T)
    y = tf.add(tf.multiply(self.h,self.t) , tf.multiply(x, (1-self.t)))

    return y

class HighwayNetwork(tf.keras.Model):
  """
  Highway Network with several layers
  """
  def __init__(self, input_size, output_size):
    super(HighwayNetwork, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.mlplayers = tf.keras.Sequential([
      tf.keras.Input(shape=(self.input_size,)),
      layers.Flatten(),
      HighwayMLP(input_size),
      layers.Dense(71, activation="relu"),
      ])
    

    # Classification layer
    self.classificationLayer = layers.Dense(self.output_size, activation='softmax')
    

  def call(self, x):
    h = self.mlplayers(x)
    out = self.classificationLayer(h)

    return out
