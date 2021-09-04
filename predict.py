import os
from argparse import ArgumentParser

from tensorflow.python.keras.backend import dtype
from model import HighwayNetwork
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    home_dir = os.getcwd()
    parser = ArgumentParser()
    parser.add_argument("--test-file-path", default='{}/data/test'.format(home_dir), type=str, required=True)
    parser.add_argument("--model-folder", default='{}/model/highway/'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=28, type=int)

    args = parser.parse_args()
    # Loading Model
    highway = tf.keras.models.load_model(args.model_folder)

    # Load test images from folder  
    def load_image(test_file_path):
        # load the image
        img = tf.keras.preprocessing.image.load_img(test_file_path, grayscale=True, target_size=(28, 28))
        # convert to array
        img = tf.keras.preprocessing.image.img_to_array(img)
        # reshape into a single sample with 3 channels
        img = img.reshape(1,784).astype("float32") / 255
        # center pixel data
        return img
    img = load_image(args.test_file_path)
    # assert img.shape == (1000,1) , 'x.shape: {}'.format(img.shape)
    predictions = highway.predict(img)   
    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(predictions))
    print('This image belongs to class: {}'.format(np.argmax(predictions), axis=1))


