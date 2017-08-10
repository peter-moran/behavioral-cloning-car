#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Peter Moran

"""
Trains a Keras model to drive the Udacity SDC-ND driving simulator based on driving data collected from that simulator.

Usage:
    `python model.py`
"""

import numpy as np
import tensorflow as tf
from keras import backend as ktf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2
from matplotlib import pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from simulator_reader import read_sim_logs, probabilistic_drop, force_gaussian

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


class VirtualSet:
    def __init__(self, sample_set, batch_size, augment=False, sidecam_angl_offfset=0.15):
        """
        Acts as an interface to sample data created by the simulator as well as augmented data, packaging them together
        as cohesive datasets (ie training set, validation set, etc.) ready for feeding into a neural network.

        :param sample_set: A dictionary created by `read_sim_log()` containing file paths to sampled images and
        simulation measurements.
        :param batch_size: Number of samples to pass to the network each call to the generator function.
        :param augment: Set True if you want this set's generator to return augmented data as well as simulator samples.
        :param sidecam_angl_offfset: Steering angle offset to be applied to simulator samples when using side cameras
        instead of center cameras. Not used if `augment` is set to False.
        """
        # Handle samples
        self.simulator_samples = sample_set
        self.n_sim_samples = len(self.simulator_samples)

        # Handle augmentation
        self.sidecam_angl_offfset = sidecam_angl_offfset
        self.n_total_samples = self.n_sim_samples
        if augment:
            self.n_total_samples *= 4

        # Batches
        self.batch_size = batch_size
        self.n_batches = self.n_total_samples / self.batch_size

    def generator_func(self):
        """
        Generator used to load images in batches as they are passed to the network, rather than loading them into
        memory all at once.
        :return: A batch of (features, labels) as numpy arrays, ready to be passed to the network.
        """
        while True:
            arg_shuffle = shuffle(range(self.n_total_samples))
            for offset in range(0, self.n_total_samples, self.batch_size):
                features = []
                labels = []
                for ndx in arg_shuffle[offset:offset + self.batch_size]:
                    sample = self.simulator_samples[ndx % self.n_sim_samples]
                    case = ndx // self.n_sim_samples
                    if case == 0:
                        # Use sample as is
                        features.append(imread(sample['img_center']))
                        labels.append(sample['angle'])
                    elif case == 1:
                        # Augment sample with reflection
                        features.append(np.fliplr(imread(sample['img_center'])))
                        labels.append(-sample['angle'])
                    elif case == 2:
                        # Augment with left camera, correcting to right
                        features.append(imread(sample['img_left']))
                        labels.append(sample['angle'] + self.sidecam_angl_offfset)
                    elif case == 3:
                        # Augment with right camera, correcting to left
                        features.append(imread(sample['img_right']))
                        labels.append(sample['angle'] - self.sidecam_angl_offfset)
                yield np.array(features), np.array(labels)


def create_model(dropout_rate=None, l2_weight=None, batch_norm=False):
    """
    Returns a Keras sequential model with normalization as specified applied.
    :param dropout_rate: Dropout rate to use on every layer. Set to `None` if you don't want to apply.
    :param l2_weight: L2 normalization weight to apply all weights. Set to `None` if you don't want to apply.
    :param batch_norm: Set `True` to apply batch normalization.
    :return: a Keras sequential model.
    """
    model = Sequential()
    if l2_weight is None:
        L2_reg = None
    else:
        L2_reg = l2(l2_weight)

    # Pre-processing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Convolution 1
    kernel_size = (5, 5)
    model.add(Conv2D(64, kernel_size, padding='same', kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # Convolution 2
    model.add(Conv2D(128, kernel_size, padding='same', kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # Convolution 3
    model.add(Conv2D(256, kernel_size, padding='same', kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())

    # Fully Connected 1
    model.add(Dense(512, kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))

    # Fully Connected 2
    model.add(Dense(128, kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    return model


if __name__ == '__main__':
    # Augmentation
    SIDECAM_OFFSET = 0.15
    VALIDATION_SPLIT = 0.4
    # Model
    DROPOUT = None
    L2_WEIGHT = None
    BATCH_NORM = False
    # Testing
    BATCH_SIZE = 40

    # Read in samples
    simulation_logs = ['./data/t1_forward/driving_log.csv', './data/t2_forward/driving_log.csv',
                       './data/t1_backwards/driving_log.csv']
    samples = read_sim_logs(simulation_logs)

    # Remove a lot of zero angles
    samples = probabilistic_drop(samples, center=0, drop_rate=.70)
    samples = force_gaussian(samples)

    # Split samples into train / test sets
    samples_train, samples_validation = train_test_split(samples, test_size=VALIDATION_SPLIT)

    # Set up generators
    train_set = VirtualSet(samples_train, batch_size=BATCH_SIZE,
                           augment=True, sidecam_angl_offfset=SIDECAM_OFFSET)
    train_generator = train_set.generator_func()
    validation_set = VirtualSet(samples_validation, batch_size=BATCH_SIZE)
    validation_generator = validation_set.generator_func()

    # Print a data summary
    print("\nTraining samples {:>12,}".format(train_set.n_total_samples))
    print("Validation samples {:>10,}".format(validation_set.n_total_samples))

    # Set up keras model
    model = create_model(dropout_rate=DROPOUT, l2_weight=L2_WEIGHT, batch_norm=BATCH_NORM)
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
