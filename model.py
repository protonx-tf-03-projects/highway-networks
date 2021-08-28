# TODO 2:  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.keras import initializers

class Dense(layers.Layer):
    def __init__(self, units, bias=0.05):
        super(Dense, self).__init__()
        self.units = units
        self.bias=bias

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        # self.b = self.add_weight(
        #     name="b", shape=(self.units,), initializer="random_normal", trainable=True,
        # )

        self.b = tf.Variable(tf.constant(self.bias, shape=self.units), name='bias', trainable=True)
        # assert self.b.shape == (784,10), 'b.shape: {}'.format(self.b.shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class HighwayMLP(layers.Layer):
  """
  Highway MLP Layer
  """
  def __init__(self, t_bias=-2.0, acti_h = tf.nn.relu, acti_t = tf.nn.tanh):
    #t_bias=-2.0, se them bias sau vao lop DENSE
    super(HighwayMLP, self).__init__()
    self.acti_h = acti_h
    self.acti_t = acti_t
    self.t = Dense(50, t_bias)
    self.h = Dense(50)
    self.x = Dense(50)

  def call(self, x, training=False):
    # Do Highway: y = H(x,WH)· T(x,WT) + x · C(x,WC).

    # assert x.shape == (784,10), 'X_T.shape: {}'.format(x.shape)
    dense_t = self.t(x, training=training)
    # assert dense_t.shape == (784,10), 'X_T.shape: {}'.format(dense_t.shape)
    dense_t = self.acti_t(dense_t)
    # assert dense_t.shape == (784,10), 'dense_t.shape: {}'.format(dense_t.shape)
    dense_h = self.h(x, training=training)
    dense_h = self.acti_h(dense_h)
    dense_x = self.x(x)


    y = tf.add(tf.multiply(dense_h, dense_t) , tf.multiply(dense_x, (1- dense_t)))
    # assert y.shape == (123,10), 'y.shape: {}'.format(y.shape)
    return y

class HighwayNetwork(tf.keras.Model):
  """
  Highway Network with several layers
  """
  def __init__(self, num_classes=10,num_of_layers=3):
    super(HighwayNetwork, self).__init__()
    # self.block1 = HighwayMLP()
    # self.block2 = HighwayMLP()
    # self.block3 = HighwayMLP()
    self.mlplayers = [ HighwayMLP() for _ in range(num_of_layers)]
    self.classifier = layers.Dense(num_classes, activation='softmax')

  def call(self, x, training=False):

    # assert x.shape == (784,10), 'X_T.shape: {}'.format(x.shape)
    # x = self.block1(x, training=training)
    # x = self.block2(x, training=training)
    # x = self.block3(x, training=training)
    for layer in self.mlplayers:
      x = layer(x, training=training)

    out = self.classifier(x)
    return out
