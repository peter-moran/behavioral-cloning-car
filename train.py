#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Peter Moran

"""
Trains a Keras model to drive the Udacity SDC-ND driving simulator based on driving data collected from that simulator.

Usage:
    `python train.py`
"""

import tensorflow as tf
from keras import backend as ktf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from augment import VirtualSet
from model import create_model
from simulator_reader import probabilistic_drop, read_sim_logs

# Set TensorFlow to allow for growth. Helps compatibility.
ktf.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)


def plot_history(fit_loss):
    """
    Creates a plot for the training and validation loss of a keras history object.
    :param fit_loss: keras history object
    """
    plt.plot(fit_loss.history['loss'])
    plt.plot(fit_loss.history['val_loss'])
    plt.title('Mean Squared Error Loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')


if __name__ == '__main__':
    #### HYPER PARAMETERS ####
    # Sample selection
    zero_angle_drop = .65  # fraction of samples with an angle of zero to drop

    # Augmentation
    flip_ratio = 1.0  # ratio of raw samples to generate mirrored samples from
    sidecam_ratio = 2.0  # ratio of raw samples to use side cameras from
    sidecam_offset = 0.13  # steering offset used for side cameras

    # Model
    dropout_rate = None  # dropout rate
    L2_weight = None  # L2 normalization weight
    batch_norm = False  # Use batch normalization?

    # Training
    validation_split = 0.4  # faction of samples to reserve for validation
    batch_size = 32  # Batch size during training
    ##########################

    # Read in samples
    simulation_logs = ['./data/t1_forward/driving_log.csv', './data/t1_backwards/driving_log.csv',
                       './data/t2_forward/driving_log.csv']
    samples = read_sim_logs(simulation_logs)

    # Remove some of the zero angle samples
    samples = probabilistic_drop(samples, 'angle', zero_angle_drop, center=0.0, margin=0.0)

    # Create datasets
    samples_train, samples_validation = train_test_split(samples, test_size=validation_split)
    train_set = VirtualSet(samples_train, batch_size, flip_ratio, sidecam_ratio, sidecam_offset)
    validation_set = VirtualSet(samples_validation, batch_size)
    train_set.simulate_angle_distribution()

    # Define generators
    train_generator = train_set.batch_generator()
    validation_generator = validation_set.batch_generator()

    # Print a data summary
    print("\nTraining samples {:>12,}".format(train_set.n_total_samples))
    print("Validation samples {:>10,}".format(validation_set.n_total_samples))

    # Set up keras model
    model = create_model(dropout_rate, L2_weight, batch_norm)
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    # Train Keras model, saving the model whenever improvements are made and stopping if loss does not improve.
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)
    checkpointer = ModelCheckpoint(filepath='./model_archive/model-{val_loss:.5f}.h5', verbose=1, save_best_only=True)
    losses = model.fit_generator(train_generator,
                                 steps_per_epoch=train_set.n_batches,
                                 validation_data=validation_generator,
                                 validation_steps=validation_set.n_batches,
                                 verbose=1,
                                 epochs=50,
                                 callbacks=[early_stopping, checkpointer])

    # Plot loss
    plot_history(losses)
    plt.ylim([0, 0.5])
    plt.show()
