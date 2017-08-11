#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Peter Moran

"""
Trains a Keras model to drive the Udacity SDC-ND driving simulator based on driving data collected from that simulator.

Usage:
    `python model.py`
"""
import random
from math import ceil

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

from simulator_reader import read_sim_logs, probabilistic_drop

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
    def __init__(self, sample_set, batch_size, augment, flip_ratio=0.0, sidecam_ratio=0.0, sidecam_offset=0.0):
        """
        Acts as an interface to sample data created by the simulator as well as augmented data, packaging them together
        as cohesive datasets (ie training set, validation set, etc.) ready for feeding into a neural network.

        :param sample_set: A dictionary created by `read_sim_log()` containing file paths to sampled images and
        simulation measurements.
        :param batch_size: Number of samples to pass to the network each call to the generator function.
        :param augment: Set True to allow data augmentation.
        :param flip_ratio: The ratio of flipped images to add to the dataset. Eg `flip_ratio=0.5` would add a flipped
        copy of half the images to the dataset. Typically ranges [0.0, 1.0].
        :param sidecam_ratio: The ratio of sidecam images to add to the dataset. Typically ranges [0.0, 2.0].
        :param sidecam_offset: Steering angle offset to be applied to simulator samples when using side cameras
        instead of center cameras. Not used if `isAugmented` is set to False.
        """
        # Handle samples
        self.raw_samples = sample_set
        self.n_raw_samples = len(self.raw_samples)

        # Handle augmentation
        self.isAugmented = augment
        if not self.isAugmented:
            assert flip_ratio == 0 and sidecam_ratio == 0 and sidecam_offset == 0, 'Not allowed without augmentation'
        self.n_flips = int(self.n_raw_samples * flip_ratio)
        self.n_sidecam = int(self.n_raw_samples * sidecam_ratio)
        self.sidecam_offset = sidecam_offset
        self.n_total_samples = self.n_raw_samples + self.n_sidecam + self.n_flips

        # Batches
        self.batch_size = batch_size
        self.n_batches = int(ceil(self.n_total_samples / self.batch_size))

    def batch_generator(self, simulate_labels=False):
        """
        Generator used to load images in batches as they are passed to the network, rather than loading them into
        memory all at once.
        :param simulate_labels: Set True to avoid loading images and only fill labels. Used for diagnostic purposes.
        :return: A batch of (features, labels) as numpy arrays, ready to be passed to the network.
        """
        # Store sample indices and image generation requests together
        FLIP_ID = -1  # request to generate a flipped image
        SIDE_ID = -2  # request to use a side camera image
        sample_map = list(range(self.n_raw_samples)) + [FLIP_ID] * self.n_flips + [
                                                                                      SIDE_ID] * self.n_sidecam  # TODO: Use more memory efficient mothod.

        while True:
            sample_map = shuffle(sample_map)
            for batch_start in range(0, len(sample_map), self.batch_size):
                features_batch = []
                labels_batch = []
                for id in sample_map[batch_start:batch_start + self.batch_size]:
                    if id >= 0:
                        # id refers to a raw image
                        sample = self.raw_samples[id]
                        f_img = sample['img_center']
                        image = imread(f_img) if not simulate_labels else f_img
                        angle = sample['angle']
                    else:
                        # Augment a random sample
                        sample = self.raw_samples[random.randint(0, self.n_raw_samples - 1)]
                        if id == FLIP_ID:
                            # Augment with reflection
                            f_img = sample['img_center']
                            image = np.fliplr(imread(f_img)) if not simulate_labels else f_img
                            angle = -sample['angle']
                        elif id == SIDE_ID:
                            # Augment with one side image
                            side = random.randint(0, 1)
                            if side == 0:
                                # Augment with left camera
                                f_img = sample['img_left']
                                image = imread(f_img) if not simulate_labels else f_img
                                angle = sample['angle'] + self.sidecam_offset
                            elif side == 1:
                                # Augment with right camera
                                f_img = sample['img_right']
                                image = imread(f_img) if not simulate_labels else f_img
                                angle = sample['angle'] - self.sidecam_offset
                    features_batch.append(image)
                    labels_batch.append(angle)
                yield np.array(features_batch), np.array(labels_batch)

    def simulate_angle_distribution(self):
        # Run generator for all samples
        batch_generator = self.batch_generator(simulate_labels=True)
        angles = []
        for n_batch in range(self.n_batches):
            features, labels = next(batch_generator)
            angles += list(labels)

        # Plot
        plt.subplot(2, 1, 1)
        plt.title('Raw Sample Distribution')
        plt.hist([s['angle'] for s in self.raw_samples], bins='auto')
        plt.xlim([-1.5, 1.5])
        plt.subplot(2, 1, 2)
        plt.title('Distribution after Augmentation (Representative)')
        plt.hist(angles, bins='auto')
        plt.xlim([-1.5, 1.5])
        plt.show()
        return angles


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
    model.add(Dense(256, kernel_regularizer=L2_reg))
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
    SIDECAM_OFFSET = 0.1
    VALIDATION_SPLIT = 0.4
    # Model
    DROPOUT = None
    L2_WEIGHT = None
    BATCH_NORM = False
    # Testing
    BATCH_SIZE = 32

    # Read in samples
    simulation_logs = ['./data/t1_forward/driving_log.csv', './data/t1_backwards/driving_log.csv',
                       './data/t2_forward/driving_log.csv', './data/t2_backwards/driving_log.csv']
    samples = read_sim_logs(simulation_logs)

    # Remove a lot of zero angles
    samples = probabilistic_drop(samples, key='angle', drop_rate=.80, center=0.0, margin=0.0)

    # Split samples into train / test sets
    samples_train, samples_validation = train_test_split(samples, test_size=VALIDATION_SPLIT)

    # Create datasets
    train_set = VirtualSet(samples_train, batch_size=BATCH_SIZE,
                           augment=True, flip_ratio=1.0, sidecam_ratio=2.0, sidecam_offset=SIDECAM_OFFSET)
    validation_set = VirtualSet(samples_validation, batch_size=BATCH_SIZE, augment=False)
    train_set.simulate_angle_distribution()

    # Define generators
    train_generator = train_set.batch_generator()
    validation_generator = validation_set.batch_generator()

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
