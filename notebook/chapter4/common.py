# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.model_selection import train_test_split


def normalize_mnist(X, y):
    return tf.cast(tf.reshape(X, [28 * 28]), "float32") / 255, y


def create_dataset(batch_size: int, validation_data_ratio: float):
    # download data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # split training data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=validation_data_ratio, shuffle=False
    )

    # make dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # normalize
    train_ds = train_ds.map(normalize_mnist)
    valid_ds = valid_ds.map(normalize_mnist)
    test_ds = test_ds.map(normalize_mnist)

    # batch
    train_ds = (
        train_ds.shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds, test_ds


def create_sequential_api_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, input_shape=(784,), activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model


def create_functional_api_model():
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


class SubclassModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation="relu")
        self.layer2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.layer1(inputs)
        outputs = self.layer2(x)
        return outputs
