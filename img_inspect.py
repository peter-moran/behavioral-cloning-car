from matplotlib.image import imread
import csv
from videofig import videofig
from simulator_reader import read_sim_logs

raw_csv_logs = ['./data/t2_forward/driving_log.csv']

# Read raw CSV
csv_data = read_sim_logs(raw_csv_logs)


# Show as a video
def redraw_fn(f, axes):
    csv_row = f + 1
    img_file = csv_data[f]['img_center']
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
