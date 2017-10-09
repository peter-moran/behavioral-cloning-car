import random
from math import ceil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from sklearn.utils import shuffle


class VirtualSet:
    def __init__(self, sample_set, batch_size,
                 flip_ratio=0.0,
                 sidecam_ratio=0.0, sidecam_offset=0.0):
        """
        Acts as an interface to sample data created by the simulator as well as augmented data, packaging them together
        as cohesive datasets (ie training set, validation set, etc.) ready for feeding into a neural network.

        :param sample_set: A dictionary created by `read_sim_log()` containing file paths to sampled images and
        simulation measurements.
        :param batch_size: Number of samples to pass to the network each call to the generator function.
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
        # Create a mapping that decides the type of image to return on each call to the generator.
        FLIP_ID = -1  # request to generate a flipped image
        SIDE_ID = -2  # request to use a side camera image
        sample_map = list(range(self.n_raw_samples)) + \
                     [FLIP_ID] * self.n_flips + \
                     [SIDE_ID] * self.n_sidecam

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
