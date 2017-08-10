import csv
import random


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
                if row is None:
                    # empty line
                    continue
                loaded_data.append({'img_center': row[0], 'img_left': row[1], 'img_right': row[2],
                                    'angle': float(row[3]), 'throttle': float(row[4]),
                                    'brake': float(row[5]), 'speed': float(row[6])})
        print('Done.')
    return loaded_data


def probabilistic_drop(samples, center, drop_rate, margin=0.01):
    drop_rate = int(drop_rate * 1000)
    return [sample for sample in samples if
            sample['angle'] > center + margin or sample['angle'] < center - margin or random.randint(0,
                                                                                                     1000) >= drop_rate]


def force_gaussian(samples, stddev=0.3, truncate=0.3):
    return [sample for sample in samples if
            abs(sample['angle']) < abs(random.gauss(0, stddev)) or abs(sample['angle']) > truncate]


def adjusted_dict_copy(d, key, f):
    ret = dict(d)
    ret[key] = f(ret[key])
    return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Load
    samples = read_sim_logs(['./data/t1_forward/driving_log.csv', './data/t1_backwards/driving_log.csv',
                             './data/t2_forward/driving_log.csv'])

    # Drop zero
    samples = probabilistic_drop(samples, center=0, drop_rate=.7)

    # Simulate sidecam
    import numpy as np

    for sample in np.copy(samples):
        samples.append(adjusted_dict_copy(sample, 'angle', lambda a: a + .15))
        samples.append(adjusted_dict_copy(sample, 'angle', lambda a: a - .15))
        samples.append(adjusted_dict_copy(sample, 'angle', lambda a: a * -1))

    gsamples = force_gaussian(samples)

    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist([s['angle'] for s in samples], bins='auto')
    plt.subplot(2, 1, 2)
    n, bins, patches = plt.hist([s['angle'] for s in gsamples], bins='auto')
    plt.show()
