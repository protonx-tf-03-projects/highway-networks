# TODO 2:
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations


class HighwayBlock(layers.Layer):
    def __init__(self, units, t_bias, acti_h, acti_t):
        super(HighwayBlock, self).__init__()
        self.units = units
        self.t_bias = t_bias
        self.acti_t = acti_t
        self.acti_h = acti_h

    def build(self, input_shape):
        self.W = self.add_weight(
            name="w",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.W_T = self.add_weight(
            name="w_T",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            name="b", shape=(self.units,), initializer="random_normal", trainable=True,
        )

        self.b_T = tf.Variable(tf.constant(self.t_bias, shape=self.units), name='bias', trainable=True)
        # assert self.b_T.shape == (50,), 'b.shape: {}'.format(self.b.shape)

    def call(self, inputs):
        h = self.acti_h(tf.matmul(inputs, self.W) + self.b)
        t = self.acti_t(tf.matmul(inputs, self.W_T) + self.b_T)
        y = tf.add(tf.multiply(h, t), tf.multiply(inputs, (1 - t)))
        return y


class HighwayNetwork(tf.keras.Model):
    """
    Highway Network with several layers
    """

    def __init__(self, t_bias=-9.0, acti_h=tf.nn.relu, acti_t=tf.nn.sigmoid, num_classes=10, num_of_layers=3):
        super(HighwayNetwork, self).__init__()
        self.projection = keras.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(50)])
        self.mlplayers = [HighwayBlock(50, t_bias=t_bias, acti_h=acti_h, acti_t=acti_t) for _ in range(num_of_layers)]

        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        X = self.projection(x)
        for layer in self.mlplayers:
            X = layer(X)
        y = self.classifier(X)
        return y