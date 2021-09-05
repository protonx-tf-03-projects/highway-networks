import tensorflow as tf
from model import HighwayNetwork
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    home_dir = os.getcwd()
    parser = ArgumentParser()
    #parser.add_argument("--test-file-path", default='{}/data/test'.format(home_dir), type=str, required=True)
    parser.add_argument("--image-size", default=28, type=int)
    parser.add_argument("--image-index", default=0, type=int)
    parser.add_argument("--model-folder", default='{}/model/highway_network/'.format(home_dir), type=str)

    # FIXME
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to Highway Network-------------------')
    print('Github: ${accout}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Predict using Highway Network for image')
    print('===========================')

    # Loading Model
    highway_network = tf.keras.models.load_model(args.model_folder)


    # Load data mnist
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Load test images from folder
    # image = tf.keras.preprocessing.image.load_img(args.test_file_path, target_size=(args.image_size, args.image_size))
    # image = tf.image.rgb_to_grayscale(
    #     image, name=None
    # )
    # input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # img = input_arr.reshape(-1,28,28,1).astype("float32") / 255

    # Normalize data

    # x_train = train_images.reshape(60000, 784).astype("float32") / 255
    img = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255

    predictions = highway_network.predict(img)
    print('---------------------Prediction Result: -------------------')
    print('Output Softmax: {}'.format(predictions[args.image_index]))
    print('This image belongs to class: {}'.format(np.argmax(predictions[args.image_index]), axis=1))

    plt.imshow(img[args.image_index])
    plt.show()
