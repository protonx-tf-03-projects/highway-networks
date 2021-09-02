import os
from argparse import ArgumentParser
from model import HighwayNetwork
import tensorflow as tf

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # # Arguments users used when running command lines
    parser.add_argument("--t-bias", default=-2.0, type=float)
    parser.add_argument("--acti-h", default=tf.nn.relu)
    parser.add_argument("--acti-t", default=tf.nn.tanh)
    parser.add_argument("--num-of-layers", default=3, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: mrmessi2512, hatruong29, quan030994, tuvu247')
    print('Email: phungngbaolong@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Highway Network model with hyper-params:') 
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    # TODO 1: Load MNIST
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    highway_network = HighwayNetwork(t_bias=args.t_bias, acti_h=args.acti_h, acti_t=args.acti_t, num_of_layers=args.num_of_layers)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # Optimizer Definition
    # All networks were optimized using SGD with momentum # Paper
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.09, nesterov=False, name="SGD")
    # Compile optimizer and loss function into model
    highway_network.compile(optimizer=sgd, loss=loss_object, metrics=['acc'])

    highway_network.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
    )

    # highway_network.summary()
