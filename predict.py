import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from data import build_dataset


if __name__ == "__main__":
    home_dir = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument("--model-folder", default='{}/model/highway/'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=28, type=int)
    parser.add_argument("--image-index", default=0, type=int)

    args = parser.parse_args()
    # Loading Model
    highway = tf.keras.models.load_model(args.model_folder)
    mnist = build_dataset()
    _, _, _, _, x_val, y_val= mnist
    predictions = highway.predict(x_val)  

    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(np.argmax(predictions[args.image_index]), axis=1))
    print('This image belongs to class: {}'.format(np.argmax(y_val[args.image_index]), axis=1))
