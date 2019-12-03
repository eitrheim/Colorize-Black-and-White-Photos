from keras.layers import Conv2D, UpSampling2D, InputLayer, Dropout, MaxPooling2D, Dense ,BatchNormalization
from keras.models import Sequential


def ModelCreator(type):
    if type == "RNN":
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

        # Finish model
        model.compile(optimizer='rmsprop', loss='mse')
    elif type == "CNN":
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(10, 10), strides=(1, 1),
                       padding='same', input_shape=(200, 200, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                       padding='same', input_shape=(200, 200, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2),
                             strides=(1, 1), padding='same'))
        model.add(Conv2D(128, kernel_size=(5, 5),
                       strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(64, input_shape=(3,), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, input_shape=(3,), activation='softmax'))
    return model