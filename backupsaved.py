import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import ReLU


def define_NN_architecture():
    wavelet_inputs = Input(shape=(248, 16, 1), name='wavelet_input')

    x = Conv2D(32, (1, 16), strides=(1, 1), padding='same')(wavelet_inputs)
   #Should these strides be higher since i want one for each row
   #Is 1x1 strides expected?
    x = ReLU()(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=None)(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=None)(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    
    #not sure if a 1X1 convolution means flatten?
    x = Flatten()(x)
    predictions = Dense(18, activation='softmax')(x)
    
  #  model = Model(inputs=[wavelet_inputs], outputs=classified_outputs)
    model = Model(inputs=[wavelet_inputs],
                  outputs=predictions)
    return model