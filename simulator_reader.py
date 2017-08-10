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

if __name__ == '__main__':
    # Load
    samples = read_sim_logs(['./data/t1_forward/driving_log.csv', './data/t1_backwards/driving_log.csv',
                             './data/t2_forward/driving_log.csv'])
