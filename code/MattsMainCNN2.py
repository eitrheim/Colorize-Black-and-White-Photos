#from keras.layers import Conv2D, UpSampling2D, InputLayer
#from keras.models import Sequential
#from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
#from Creating_Matrix import MatrixCreator
from Matrix_Creator import CreateMatrix
from Convert_Color import ColConvert
from Model_Creation import ModelCreator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#from keras.utils import plot_model
#from keras import backend as K


os.environ['KMP_DUPLICATE_LIB_OK']='True'

batch_size = 10
epoch_num = 100

# Get images

matrix = CreateMatrix()
X, Y = ColConvert(matrix, "rbg2lab")

print("Fitting the Model")

# Building the neural network

model = ModelCreator("CNN2")

# Finish model
model.compile(optimizer='rmsprop', loss='mse')

#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                               cooldown=0,
#                               patience=5,
#                               verbose=1,
#                               min_lr=0.5e-6)

earlystopping = EarlyStopping(monitor='loss',
                              patience=7)

callbacks = [earlystopping]

model.fit(x=X,
          y=Y,
          epochs=epoch_num,
          callbacks=callbacks)
print(model.evaluate(X, Y, batch_size=batch_size))



testMatrix = CreateMatrix(test = True)
testX, testY = ColConvert(testMatrix, "rbg2lab")

#imsave("../TestPhoto/img_predict.jpg", testX)



output = model.predict(testX)
output *= 128
# Output colorizations

cur = np.zeros((testX.shape[1], testX.shape[2], 3))
cur[:,:,0] = testX[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("../TestPhoto/img_predict.jpg", lab2rgb(cur))
imsave("../TestPhoto/img_gray_version.jpg", rgb2gray(lab2rgb(cur)))

