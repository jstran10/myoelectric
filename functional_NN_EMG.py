import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import ReLU


def define_NN_architecture():
    wavelet_inputs = Input(shape=(248, 16, 1), name='wavelet_input')

    rms_inputs = Input(shape=(16, ), name='rms_input')
    RMS_out = BatchNormalization(
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones'
                        )(rms_inputs)

    x = Conv2D(
                        32,
                        (3, 3),
                        padding='same',
                        )(wavelet_inputs)

    x = BatchNormalization(
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones'
                        )(x)
    x = ReLU()(x)

    x_parallel = x

    x_parallel = MaxPooling2D((2, 2), padding='same')(x_parallel)

    x = Conv2D(
                        32,
                        (3, 3),
                        padding='same',
                        )(x)
    x = BatchNormalization(
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones'
                        )(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(
                        32,
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        )(x)

    x = keras.layers.concatenate([x, x_parallel], axis=3)

    x_parallel = x

    x_parallel = MaxPooling2D((2, 2), padding='same')(x_parallel)

    x = BatchNormalization(
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones'
                        )(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(
                        32,
                        (3, 3),
                        padding='same',
                        )(x)

    x = BatchNormalization(
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones'
                        )(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(
                        32,
                        (3, 3),
                        strides=(2, 2),
                        padding='same',
                        )(x)

    x = keras.layers.concatenate([x, x_parallel], axis=3)

    x = Flatten()(x)
    x = BatchNormalization(
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer='zeros',
                        gamma_initializer='ones',
                        moving_mean_initializer='zeros',
                        moving_variance_initializer='ones'
                        )(x)
    wavelet_out = ReLU()(x)

    combined_inputs = keras.layers.concatenate(
                [RMS_out, wavelet_out]
            )

    x = Dense(120,
              activation='relu'
              )(combined_inputs)

    x = Dropout(0.5)(x)

    predictions = Dense(18,
                        activation='softmax'
                        )(x)

    model = Model(inputs=[wavelet_inputs, rms_inputs],
                  outputs=predictions)
    return model
