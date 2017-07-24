import csv
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers import Lambda, MaxPooling2D, Flatten, Dense
from keras import backend as K
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Fix tf bug
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

# Options
simulation_logs = ['./data/t1_forward/driving_log.csv']


def read_sim_logs(csv_paths):
    """
    Reads each `.csv` file and stores the image file paths and measurement values to a list of dictionaries.
    :param csv_paths: list of file paths to CSV files created by the simulator.
    :return: list of dictionaries containing image files and measurements from the simulator at each sample.
    """
    loaded_data = []
    for path in csv_paths:
        print('Loading data from "{}"...'.format(path))
        with open(path, 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                loaded_data.append({'img_center': row[0], 'img_left': row[1], 'img_right': row[2],
                                    'angle': float(row[3]), 'throttle': float(row[4]),
                                    'brake': float(row[5]), 'speed': float(row[6])})
        print('Done.')
    return loaded_data


def plot_history(losses):
    plt.plot(losses.history['loss'])
    plt.plot(losses.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')


class VirtualSet:
    def __init__(self, sample_set, batch_size, augment=False):
        """
        Acts as an interface to both real sampled data and augmented data for datasets passed through the network
        (ie training set, validation set, etc.)

        :param sample_set: A dictionary created by `read_sim_log()` containing file paths to sampled images and
        simulation measurements. This will be the raw data for the training, validation, or test set.
        :param batch_size: Number of samples to pass to the network each call to the generator function.
        """
        self.raw_samples = sample_set
        self.n_raw_samples = len(self.raw_samples)
        self.batch_size = batch_size
        self.n_samples = self.n_raw_samples
        if augment:
            self.n_samples *= 2
        self.n_batches = self.n_samples / self.batch_size

    def generator_func(self):
        """
        Generator used to load images in batches as they are passed to the network, rather than
        loading them into memory all at once.
        :return: features and associated labels, ready to be passed to the network.
        """
        while True:
            arg_shuffle = shuffle(range(self.n_samples))
            for offset in range(0, self.n_samples, self.batch_size):
                features = []
                labels = []
                for ndx in arg_shuffle[offset:offset+self.batch_size]:
                    sample = self.raw_samples[ndx % self.n_raw_samples]
                    image = imread(sample['img_center'])
                    angle = sample['angle']
                    if ndx < self.n_raw_samples:
                        # Use sample as is
                        features.append(image)
                        labels.append(angle)
                    elif ndx >= self.n_raw_samples:
                        # Augment sample
                        features.append(np.fliplr(image))
                        labels.append(-angle)
                yield np.array(features), np.array(labels)


def create_model():
    model = Sequential()

    # Pre-processing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # VGG inspired structure
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    # Read in samples
    simulator_samples = read_sim_logs(simulation_logs)

    # Split samples into train / test sets
    samples_train, samples_validation = train_test_split(simulator_samples, test_size=0.3)

    # Set up generators
    train_set = VirtualSet(samples_train, batch_size=32, augment=True)
    train_generator = train_set.generator_func()
    validation_set = VirtualSet(samples_validation, batch_size=32)
    validation_generator = validation_set.generator_func()

    # Train keras model
    model = create_model()
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    losses = model.fit_generator(train_generator,
                                 steps_per_epoch=train_set.n_batches,
                                 validation_data=validation_generator,
                                 validation_steps=validation_set.n_batches,
                                 epochs=6)
    model.save('model.h5')

    # Plot loss
    plot_history(losses)
    plt.ylim([0, .005])
    plt.show()
