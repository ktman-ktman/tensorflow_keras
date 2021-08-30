# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.model_selection import train_test_split


def normalize_cifar10(X, y):
    return tf.math.divide(tf.cast(X, "float32"), 255), y


def make_cifar10_dataset(batch_size: int) -> tuple:

    # download data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # split training data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    # make tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # normalize data
    train_ds = train_ds.map(normalize_cifar10)
    valid_ds = valid_ds.map(normalize_cifar10)
    test_ds = test_ds.map(normalize_cifar10)

    # batch
    train_ds = (
        train_ds.shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    valid_ds = valid_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, valid_ds, test_ds


def create_sequential_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32,
            input_shape=(32, 32, 3),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(
        tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model
