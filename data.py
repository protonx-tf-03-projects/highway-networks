import tensorflow as tf

def build_dataset():
    # load data mnist
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    datasets = []
    # Normalize data
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    train_num = len(x_train)

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    
    num_digits = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_digits)
    y_test = tf.keras.utils.to_categorical(y_test, num_digits)
    y_val = tf.keras.utils.to_categorical(y_val, num_digits)

    return x_train, y_train, x_test, y_test, x_val, y_val
