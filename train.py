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
    home_dir = os.getcwd()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--number-of-layers", default=10, type=int)
    parser.add_argument("--model-folder", default='{}/model/highway_network/'.format(home_dir), type=str)

    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${accout}')
    print('Email: tranquan030894@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Highway Network model with hyper-params:') # FIXME
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # FIXME
    # Do Prediction 

    # batch_size = 128
    # highway_number = 10
    # epochs = 5
    # log_interval = 10


    # TODO 1: Load MNIST

    number_class = 10
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    # Normalize data

    x_train = train_images.reshape(60000, 784).astype("float32") / 255
    x_test = test_images.reshape(10000, 784).astype("float32") / 255



    model = HighwayNetwork(num_of_layers = args.number_of_layers)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # Optimizer Definition
    # All networks were optimized using SGD with momentum # Paper
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.09, nesterov=False, name="SGD")
    # Compile optimizer and loss function into model
    model.compile(optimizer=sgd, loss=loss_object, metrics=['acc'])

    # Do training model
    model.fit(
        x_train, train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, test_labels),
    )

    #saving model
    model.save(args.model_folder)

    # TODO 3: Custom training

    

