from keras.layers import Activation, BatchNormalization, Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, \
    MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2


def create_model(dropout_rate=None, l2_weight=None, batch_norm=False):
    """
    Returns a Keras sequential model with normalization as specified applied.
    :param dropout_rate: Dropout rate to use on every layer. Set to `None` if you don't want to apply.
    :param l2_weight: L2 normalization weight to apply all weights. Set to `None` if you don't want to apply.
    :param batch_norm: Set `True` to apply batch normalization.
    :return: a Keras sequential model.
    """
    model = Sequential()
    if l2_weight is None:
        L2_reg = None
    else:
        L2_reg = l2(l2_weight)

    # Pre-processing
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Convolution 1
    kernel_size = (5, 5)
    model.add(Conv2D(64, kernel_size, padding='same', kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # Convolution 2
    model.add(Conv2D(128, kernel_size, padding='same', kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # Convolution 3
    model.add(Conv2D(256, kernel_size, padding='same', kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())

    # Fully Connected 1
    model.add(Dense(512, kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))

    # Fully Connected 2
    model.add(Dense(256, kernel_regularizer=L2_reg))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('elu'))
    if dropout_rate is not None:
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    return model