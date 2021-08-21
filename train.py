
import os
from argparse import ArgumentParser

from tensorflow.python.keras.layers.core import Flatten
from model import HighwayNetwork
# from model import HighwayMLP

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# def accuracy_function(real, pred):
#     pass


# def train_step(image, label):
#     pass


# def val_step():
#     pass


# @tf.function
# def train():
#     pass
    

if __name__ == "__main__":
    # parser = ArgumentParser()
    
    # # FIXME
    # # Arguments users used when running command lines
    # parser.add_argument("--batch-size", default=64, type=int)
    # parser.add_argument("--epochs", default=1000, type=int)

    # home_dir = os.getcwd()
    # args = parser.parse_args()

    # # FIXME
    # # Project Description

    # print('---------------------Welcome to ${name}-------------------')
    # print('Github: ${accout}')
    # print('Email: ${email}')
    # print('---------------------------------------------------------------------')
    # print('Training ${name} model with hyper-params:') # FIXME
    # print('===========================')
    
    # FIXME
    # Do Prediction 

    batch_size = 128
    highway_number = 5
    epochs = 10
    log_interval = 10


    # TODO 1: Load MNIST

    img_width, img_height = 28, 28
    # img_size = img_width * img_height
    number_class = 10
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    
    x_train = train_images.reshape(60000, 784).astype("float32") / 255
    x_test = test_images.reshape(10000, 784).astype("float32") / 255


    input_size=x_train.shape[1]
    highway_network = HighwayNetwork(input_size,number_class)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # Optimizer Definition
    # All networks were optimized using SGD with momentum # Paper
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.09, nesterov=False, name="SGD")
    # Compile optimizer and loss function into model
    highway_network.compile(optimizer='SGD', loss=loss_object, metrics=['acc'])

    highway_network.fit(
        x_train, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, test_labels),
    )
