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


def create_functional_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


class SubclassModelCreator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv2d_1 = tf.keras.layers.Conv2D(
            32,
            input_shape=(32, 32, 3),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
        self.conv2d_2 = tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.max_pooling_2d_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2d_3 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv2d_4 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.max_pooling_2d_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout_1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.max_pooling_2d_1(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.max_pooling_2d_2(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_2(x)
        x = self.dense_2(x)
        return x
