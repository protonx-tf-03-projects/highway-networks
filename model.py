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
    self.acti_h =None
    self.acti_t = None

    # TODO 

    # Dense H
    self.h = None
    # Dense T
    self.t =None


    pass

  def call(self, x):
    # Do Highway: y = H(x,WH)· T(x,WT) + x · C(x,WC).
    y = None


class HighwayNetwork(tf.keras.Model): 
  """
  Highway Network with several layers
  """
  def __init__(self, input_size, output_size):
    super(HighwayNetwork, self).__init__()
    self.mlplayers = [
      # TO DO
    ]

    # Classification layer

  def call(self, x):
    # Run input on these mlp layers
    

    # pass output to classification layer



model = HighwayNetwork(784, 10)
