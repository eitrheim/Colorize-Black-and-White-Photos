from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
from Creating_Matrix import MatrixCreator
from Matrix_Creator import CreateMatrix
from Convert_Color import ColConvert


os.environ['KMP_DUPLICATE_LIB_OK']='True'



# Get images

matrix = CreateMatrix()
X, Y = ColConvert(matrix, "rbg2lab")

print("Fitting the Model")

# Building the neural network
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

model.fit(x=X, y=Y, batch_size=1, epochs=1)

testMatrix = CreateMatrix(test = True)
testX, testY = ColConvert(testMatrix, "rbg2lab")


print(model.evaluate(X, Y, batch_size=1))
output = model.predict(testX)
output *= 128
# Output colorizations

cur = np.zeros((testX.shape[0], testX.shape[1])) #TODO, THIS IS FAILING
cur[:,:,0] = testX[:,:,0]
cur[:,:,1:] = output[0]
imsave("../TestPhoto/img_predict.jpg", lab2rgb(cur))
imsave("../TestPhoto/img_gray_version.jpg", rgb2gray(lab2rgb(cur)))

