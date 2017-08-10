# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## The Project
The goals of this project were to:
* Use a vehicle simulator to collect data (dashboard video and steering angles) of good driving.
* Design, train, and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around both the basic
and difficult track in the simulator for at least one lap, without ever
moving / falling / touching off the track.

## Dependencies
If you wish to run all aspects of this project, you will require:

* Python 3
* Keras 2
* TensorFlow, numpy, matplotlib, sklearn, and moviepy.
    * All of these dependencies are easily provided via the [Udacity CarND Term 1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit).
* The [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim).
* The python script [videofig.py](https://github.com/bilylee/videofig).

## Design Overview
### Creation of the Training Set

Getting good driving examples for training is very important. If you put
garbage in, you'll get garbage out, so we shoot to get the best driving
data possible.

The training set was built by driving around the vehicle simulator and sampling
dashboard camera video and steering angles at each moment. A video game controller was used
for precise control of the driving. This allowed me to make smooth turns
and have steering data that accurately reflects the ideal driving angle
at any given moment.

**Modeling Driving Recovery**

Unfortunately, data recorded from people driving does not often show
examples of having to re-center the car if it drifts towards the edge of
the track. In order to train for this behavior, the training set was
augmented by using additional photos from cameras on the left and right side
of the car dashboard as obtained during simulation. These images were used
as if they were captured at the center of the car and accordingly a small,
correcting steering angle offset is applied. This, in effect, trains the
car to correct to the center of the road in such situations where it is
off center without a human driver having to manually show such behavior.
Using this approach saw the greatest amount of improvement verses every
other tactic used in this project and is essential to success.

The training set was also augmented with mirror images of every single center image
sample. Overall, 25% of the training data was raw sampled center images, 25% were flipped
versions of those center images, and 50% were the side camera images with steering offsets.
In the future, it may be helpful to select for specific samples so that
each driving angle is represented by an equal distribution of samples,
but this was not needed to achieve the goals of the project.

### Model Architecture

The model consists of a series of three sets of 5x5 convolution, relu activation,
and 3x3 pooling, which are then followed by two fully connected layers
and then the output layer. In all, the model has 3,979,137 training parameters.

The network uses 3x3 max pooling in order to keep the model
small enough to run on a 2GB graphics card. Performance was still great despite
the potential loss of information in these layers.

Also, rather than convolving until we have a single dimensional feature map, as may be expected
and was done by NVIDIA in their paper "[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)",
the final convolutional layer (i.e. just before flattening for passing to the fully  connected layers)
returns feature maps with dimensions 2x11. The reasoning behind this is to retain some
locality of features. Convolution is great for being location independent, but in this case the
position of features very well could be important to optimization, and thus making some vertical
position information available to the network is beneficial.


**Detailed Model Summary**
```
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 65, 320, 64)       4864
_________________________________________________________________
activation_1 (Activation)    (None, 65, 320, 64)       0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 106, 64)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 106, 128)      204928
_________________________________________________________________
activation_2 (Activation)    (None, 21, 106, 128)      0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 35, 128)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 35, 256)        819456
_________________________________________________________________
activation_3 (Activation)    (None, 7, 35, 256)        0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 11, 256)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 5632)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               2884096
_________________________________________________________________
activation_4 (Activation)    (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 128)               65664
_________________________________________________________________
activation_5 (Activation)    (None, 128)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 129
=================================================================
```

## Details About Files In This Directory

### `model.py`

This file, `model.py`, creates a trained keras model that predicts
steering angle based on dashboard camera images. To run, simply use the
command:

```sh
python model.py
```

To make things easier, training will  terminate if validation loss fails to
improve after too many epochs, and any time validation performance is improved,
the model  will be saved to `./model_archive/model-<loss>.h5` (with the loss
noted in the filename, as shown).

To keep the repository clean, the folder `./model_archive/` is not tracked. If
better performance is achieved than in the official model, `model.h5`, then
`model.h5` file should be replaced.

### `drive.py`

Once a model has been created, it can be used in the simulator with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

**Saving a video**

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `simulator_reader.py`
Contains a function for cleanly reading the contents of the simulator files. This has been factored out to
make imports cleaner.

### `img_inspect.py`


### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.