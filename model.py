# TODO 2:
import tensorflow as tf
import numpy as np

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

    w_init = tf.random_normal_initializer()


    self.W_H = tf.Variable(
      initial_value = w_init(shape=(self.size,self.size), dtype="float32" )
      ,trainable=True
    )
    self.b_H = tf.Variable(
      tf.constant(0.1, shape=(self.size,1))
      , dtype="float32"
      , trainable=True
    )

    self.W_T = tf.Variable(
      initial_value = w_init(shape=(self.size,self.size), dtype="float32" )
      , trainable=True
    )
    self.b_T = tf.Variable(
      tf.constant(self.bias_init, shape=(self.size,1))
      , dtype="float32"
      , trainable=True
    )


  def call(self, x):
    # Do Highway: y = H(x,WH)· T(x,WT) + x · C(x,WC).

    # TODO
    # Dense H
    self.h = self.acti_h(tf.matmul(x, self.W_H) + self.b_H)

    # Dense T

    self.t = self.acti_t(tf.matmul(x, self.W_T) + self.b_T)

    y = tf.add(tf.multiply(self.h,self.t) , tf.multiply(self.input, (1-self.t)))

    return y


class HighwayNetwork(tf.keras.Model):
  """
  Highway Network with several layers
  """
  def __init__(self, input_size, output_size):
    super(HighwayNetwork, self).__init__()
    #self.projection = tf.keras.layers.Dense(input_size)
    self.mlplayers = [
      # TO DO
      HighwayMLP(input_size)
    ]

    # Classification layer
    self.classificationLayer = tf.keras.Sequential([
      #tf.keras.layers.GlobalAveragePooling1D(),
      #tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(output_size, activation='softmax')
    ])

  def call(self, x):
    # Run input on these mlp layers

    #x = self.projection(x)
    for mlplayers in self.mlplayers:
      y = mlplayers(x)


    # pass output to classification layer
    out = self.classificationLayer(y)

    return out

model = HighwayNetwork(784, 10)