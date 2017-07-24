from matplotlib.image import imread
import csv
from videofig import videofig

raw_csv_logs = ['./data/t1_forward/driving_log.csv']

# Read raw CSV
csv_data = []
for csv_path in raw_csv_logs:
    print('Loading paths from "{}"...'.format(csv_path))
    with open(csv_path, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            csv_data.append({'file_img_center': row[0], 'file_image_left': row[1], 'file_img_right': row[2],
                             'steering': row[3], 'throttle': row[4], 'brake': row[5], 'speed': row[6]})
    print('Done.')


# Show as a video
def redraw_fn(f, axes):
    csv_row = f + 1
    img_file = csv_data[f]['file_img_center']
    img = imread(img_file)
    if not redraw_fn.initialized:
        redraw_fn.im = axes.imshow(img, animated=True)
        axes.set_title(csv_row)
        redraw_fn.initialized = True
    else:
        redraw_fn.im.set_array(img)
        axes.set_title(csv_row)
redraw_fn.initialized = False

videofig(len(csv_data), redraw_fn, play_fps=30)