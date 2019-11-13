from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
from Creating_Matrix import MatrixCreator
from Matrix_Creator import CreateMatrix


os.environ['KMP_DUPLICATE_LIB_OK']='True'



# Get images

matrix = CreateMatrix():
for i in range(matrix.shape[0]):
    image = matrix[i]
    #image = np.transpose(image)
    if i == 0:
        X = rgb2lab(1.0/255*image)[:,:,0]
        Y = rgb2lab(1.0/255*image)[:,:,1:]
        Y = Y / 128
        X = X.reshape(1, 200, 200, 1)
        Y = Y.reshape(1, 200, 200, 2)
    else:
        tempX = rgb2lab(1.0/255*image)[:,:,0]
        tempY = rgb2lab(1.0/255*image)[:,:,1:]
        tempY = tempY / 128
        tempX = tempX.reshape(1, 200, 200, 1)
        tempY = tempY.reshape(1, 200, 200, 2)
        X = np.vstack([X, tempX])
        Y = np.vstack([Y, tempY])
print("Finished Splitting Data!")

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


testMatrix = MatrixCreator(test = True)
#testMatrix = np.transpose(testMatrix)

testX = rgb2lab(1.0 / 255 * testMatrix)[:, :, 0]
testY = rgb2lab(1.0 / 255 * testMatrix)[:, :, 1:]
testY = testY / 128
testX = testX.reshape(1, 200, 200, 1)
testY = testY.reshape(1, 200, 200, 2)

print(model.evaluate(X, Y, batch_size=1))
output = model.predict(testX)
output *= 128
# Output colorizations
cur = np.zeros((testX.shape[0], testX.shape[1], 3))
print(testX[:,:,0].shape)
print(np.transpose(testX)[:,:,0].shape)
cur[:,:,0] = testX[:,:,0]
cur[:,:,1:] = output[0]
imsave("../TestPhoto/img_predict.jpg", lab2rgb(cur))
imsave("../TestPhoto/img_gray_version.jpg", rgb2gray(lab2rgb(cur)))


