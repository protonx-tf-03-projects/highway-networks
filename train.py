import os
from argparse import ArgumentParser
from model import HighwayNetwork
from model import HighwayMLP
from tensorflow.keras import backend as K

import tensorflow as tf

def accuracy_function(real, pred):
    pass


def train_step(image, label):
    pass


def val_step():
    pass


@tf.function
def train():
    pass
    

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')
    
    # FIXME
    # Do Prediction 

    batch_size = 64
    highway_number = 5
    epochs = 10
    log_interval = 10


    # TODO 1: Load MNIST

    img_width, img_height = 28, 28
    img_size = img_width * img_height
    number_class = 10
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    # Normalize data
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    def flatten_images(images):

        batch_size = images.shape[0]
        image_vector_size = images.shape[1] * images.shape[2]
        flattened_images = images.flatten().reshape(batch_size, image_vector_size)

        return flattened_images
    x_train = flatten_images(train_images)
    x_test = flatten_images(test_images)


    highway_network = HighwayNetwork(img_size,number_class)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # Optimizer Definition
    # All networks were optimized using SGD with momentum # Paper
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.09, nesterov=False, name="SGD")
    # Compile optimizer and loss function into model
    highway_network.compile(optimizer=sgd, loss=loss_object, metrics=['acc'])

    highway_network.fit(
        x_train, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        #validation_data=(x_test, test_labels),
    )


    # TODO 3: Custom training

    

