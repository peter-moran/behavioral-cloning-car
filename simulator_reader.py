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


def probabilistic_drop(samples, key, drop_rate, center, margin=0.01):
    """
    Removes random selection of entries in `samples` for every entry where the value stored at `key` is within a margin of center.
    Ex: To remove 60% of samples that have an angle within 0.1 of zero
        probabilistic_drop(samples, 'angle', 0.6, 0.0, 0.1)
    :return:
    """
    assert drop_rate >= 0 and drop_rate <= 1.0, 'drop rate must be a fraction'
    assert margin >= 0, 'margin must be non-negative'
    drop_rate = int(drop_rate * 1000)
    return [sample for sample in samples if
            sample[key] > center + margin or sample[key] < center - margin or random.randint(0, 1000) >= drop_rate]
