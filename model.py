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


def feature_label_generator(samples, batch_size=32):
    """
    Generator used to load images in batches as they are passed to the network, rather than
    loading them into memory all at once.
    :param samples: a dictionary containing image files and simulation measurements.
    :param batch_size:
    :return:
    """
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for batch_start in range(0, num_samples, batch_size):
            batch = samples[batch_start:batch_start+batch_size]

            features = np.array([imread(sample['img_center']) for sample in batch])
            labels = np.array([sample['angle'] for sample in batch])

            yield features, labels


def create_model():
    model = Sequential()

    # Preprocessing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    # VGG inspired structure
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
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
    BATCH_SIZE = 32
    train_generator = feature_label_generator(samples_train, batch_size=BATCH_SIZE)
    validation_generator = feature_label_generator(samples_validation, batch_size=BATCH_SIZE)

    # Train keras model
    model = create_model()
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    losses = model.fit_generator(train_generator,
                                 steps_per_epoch=len(samples_train)/BATCH_SIZE,
                                 validation_data=validation_generator,
                                 validation_steps=len(samples_validation)/BATCH_SIZE,
                                 epochs=10)
    model.save('model.h5')

    # Plot loss
    plt.figure()
    plot_history(losses)
    plt.figure()
    plot_history(losses)
    plt.ylim([0,.005])
    plt.show()