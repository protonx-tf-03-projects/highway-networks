import os
from argparse import ArgumentParser
from model import HighwayNetwork
from tensorflow.keras import backend as K

import tensorflow as tf

    
if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--t-bias", default=-9.0, type=float)
    parser.add_argument("--acti-h", default=tf.nn.relu)
    parser.add_argument("--acti-t", default=tf.nn.sigmoid)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-layers", default=15, type=int)
    parser.add_argument("--model-folder", default='{}/model/highway_network/'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=28, type=int)
    parser.add_argument("--image-channels", default=1, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)

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

    # TODO 1: Load MNIST

    number_class = 10
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    # Normalize data

    x_train = train_images.astype("float32")/ 255
    x_test = test_images.astype("float32") /255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = HighwayNetwork(t_bias=args.t_bias,
                           acti_h=args.acti_h,
                           acti_t=args.acti_t,
                           num_of_layers=args.number_of_layers)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # Optimizer Definition
    # All networks were optimized using SGD with momentum # Paper
    sgd = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.09, nesterov=False, name="SGD")
    # Compile optimizer and loss function into model
    model.compile(optimizer=sgd, loss=loss_object, metrics=['acc'])

    # Do training model
    model.fit(
        x_train, train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data= (x_test, test_labels),
    )
    #
    #saving model
    model.save(args.model_folder)

    # TODO 3: Custom training

    

