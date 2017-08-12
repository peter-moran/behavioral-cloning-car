import csv
import random

import matplotlib.pyplot as plt


def read_sim_logs(csv_paths):
    """
    Reads each `.csv` file and stores the image file paths and measurement values to a list of dictionaries.
    :param csv_paths: list of file paths to CSV files created by the simulator.
    :return: list of dictionaries containing image files and measurements from the simulator at each sample.
    """
    loaded_data = []
    if not isinstance(csv_paths, list):
        csv_paths = [csv_paths]
    for i_path, path in enumerate(csv_paths):
        print('Loading data from "{}"...'.format(path))
        with open(path, 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row is None:
                    # empty line
                    continue
                loaded_data.append({'img_center': row[0], 'img_left': row[1], 'img_right': row[2],
                                    'angle': float(row[3]), 'throttle': float(row[4]),
                                    'brake': float(row[5]), 'speed': float(row[6]), 'origin': i_path})
        print('Done.')
    return loaded_data


def probabilistic_drop(samples, key, drop_rate, center, margin=0.01):
    """
    Removes random selection of entries in `samples` for every entry where the value stored at `key` is within a margin of center.
    Ex: To remove 60% of samples that have an angle within 0.1 of zero
        probabilistic_drop(samples, 'angle', 0.6, 0.0, 0.1)
    :return:
    """
    assert 0 <= drop_rate <= 1.0, 'drop rate must be a fraction'
    assert margin >= 0, 'margin must be non-negative'
    drop_rate = int(drop_rate * 1000)
    return [sample for sample in samples if
            sample[key] > center + margin or sample[key] < center - margin or random.randint(0, 1000) >= drop_rate]


def compare_sample_origin(samples, names):
    # Read in data separably
    n_samples = [0] * len(names)
    for sample in samples:
        n_samples[sample['origin']] += 1

    # Plot
    bars = plt.bar(range(len(n_samples)), n_samples)
    plt.xlabel('Sample Set')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Sizes')
    plt.xticks(range(len(n_samples)), names)

    # Add count lables to each bar.
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 '%d' % int(height),
                 ha='center', va='bottom')
    plt.show()


if __name__ == '__main__':
    simulation_logs = ['./data/t1_forward/driving_log.csv', './data/t1_backwards/driving_log.csv',
                       './data/t2_forward/driving_log.csv', './data/t2_backwards/driving_log.csv']
    samples = read_sim_logs(simulation_logs)

    # Remove a lot of zero angles
    samples = probabilistic_drop(samples, key='angle', drop_rate=.80, center=0.0, margin=0.0)

    compare_sample_origin(samples, ('T1F', 'T1B', 'T2F', 'T2B'))
