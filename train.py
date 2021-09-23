import os
from argparse import ArgumentParser
from model import HighwayNetwork
import tensorflow as tf
from data import build_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()
    # # Arguments users used when running command lines
    parser.add_argument("--t-bias", default=-2.0, type=float)
    parser.add_argument("--acti-h", default=tf.nn.relu)
    parser.add_argument("--acti-t", default=tf.nn.sigmoid)
    parser.add_argument("--num-of-layers", default=3, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--model-folder", default='{}/model/highway_network/'.format(home_dir), type=str)

    args = parser.parse_args()

    # # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: pnbl-123, hatruong29, quan030994')
    print('---------------------------------------------------------------------')
    print('Training Highway Network model with hyper-params:') 
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')

    mnist = build_dataset()
    x_train, y_train, x_test, y_test, x_val, y_val= mnist
    
    highway_network = HighwayNetwork(t_bias=args.t_bias, acti_h=args.acti_h, acti_t=args.acti_t, num_of_layers=args.num_of_layers)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # Optimizer Definition
    # All networks were optimized using SGD with momentum # Paper
    sgd = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.09, nesterov=False, name="SGD")
    # Compile optimizer and loss function into model
    highway_network.compile(optimizer=sgd, loss=loss_object, metrics=['acc'])

    highway_network.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
    )
    highway_network.evaluate(x_val, y_val)
    highway_network.save(args.model_folder)
    # highway_network.summary()
