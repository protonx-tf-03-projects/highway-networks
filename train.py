import os
from argparse import ArgumentParser
from model import HighwayNetwork

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


    # TODO 3: Custom training

    

