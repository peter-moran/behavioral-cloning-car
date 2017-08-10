"""
Reads simulator CSV data files and plays back the samples as a video, with timestamps for each frame.
This makes it simple to remove bad segments of driving or trim the beginning and ends of logs.

Depends on `videofig.py` by Bily Lee
For latest version, go to https://github.com/bilylee/videofig

Usage to view and scrobble through the images from ./data/t1_forward/driving_log.csv
    Run `python img_inspect.py ./data/t1_forward/driving_log.csv`
    Use 'Enter' to play/pause. Arrow keys to go frame by frame.
    The frame number appears on the top of the screen and should correspond to the row of that image in the csv file.
"""
import argparse

from matplotlib.image import imread

from simulator_reader import read_sim_logs
from videofig import videofig

# Read arguments
parser = argparse.ArgumentParser(description='Scrobble through driving images.')
parser.add_argument(
    'driving_log',
    type=str,
    default='',
    help='Path to driving_log.csv.'
)
args = parser.parse_args()

# Read raw CSV
csv_data = read_sim_logs([args.driving_log])


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
