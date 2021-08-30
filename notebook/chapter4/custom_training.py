# -*- coding: utf-8 -*-

import tensorflow as tf


def train_step(
    model, optimizer, x_train, y_train, loss_object, train_loss, train_accuracy
):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    if train_accuracy:
        train_accuracy(y_train, predictions)


def test_step(model, x_valid, y_valid, loss_object, test_loss, test_accuracy):
    predictions = model(x_valid)
    loss = loss_object(y_valid, predictions)

    test_loss(loss)
    if test_accuracy:
        test_accuracy(y_valid, predictions)


def train_and_test(
    model,
    loss_object,
    optimizer,
    train_dataset,
    test_dataset,
    epochs,
    train_loss,
    test_loss,
    train_accuracy,
    test_accuracy,
    train_summary_writer,
    test_summary_writer,
):
    for epoch in range(epochs):
        for (x_train, y_train) in train_dataset:
            train_step(
                model,
                optimizer,
                x_train,
                y_train,
                loss_object,
                train_loss,
                train_accuracy,
            )
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_dataset:
            test_step(model, x_test, y_test, loss_object, test_loss, test_accuracy)

        with test_summary_writer.as_default():
            tf.summary.scalar("loss", test_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)

        print(
            f"Epoch {epoch+1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}"
        )

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
