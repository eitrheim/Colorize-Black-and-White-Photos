# Deep Learning and Image Recognition - Colorization Autoencoder
# The autoencoder is trained with grayscale images as input and colored images as output.
# Colorization autoencoder can be treated like the opposite of denoising autoencoder.
# Instead of removing noise, colorization adds noise (color) to the grayscale image.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras import backend as K

from skimage import io
from skimage.transform import resize
from skimage.transform import rotate, rescale

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random


start_time = time.time()

############### read images and prepare for model ###############
repo_path = os.path.dirname(os.getcwd())
file_name = [f for f in os.listdir(repo_path + '/images') if f.endswith(('.jpg', '.JPG'))]

# read in colored pictures, make same size, and rotate for data augmentation & increase training size
for img_id in file_name:
    color = io.imread(os.path.join(repo_path + '/images', img_id))

    # make the image smaller and the same sized square
    resize_size = 2**7  # needs to be sized to 2 to the power of x (2**7 == 128)
    color = resize(color, (resize_size, resize_size))

    # making the training array with the image + it rotated
    if 'colored' not in globals():
        colored = [np.array(color)]
        # colored.append(np.array(rotate(color, random.randint(0, 360), resize=False)))
    else:
        colored.append(np.array(color))
        # colored.append(np.array(rotate(color, random.randint(0, 360), resize=False)))

    if np.array(color).shape != (128, 128, 3):
        print(np.array(color).shape)
        print(img_id)

colored = np.array(colored)
print('Shape:', colored.shape)

# create train/test sets
# train/test that is roughly 75/25 (not exact due to duplicates in this method)
x_test = sorted(list(set(np.random.randint(0, colored.shape[0], size=int(colored.shape[0] / 3.5)))))
print('Test size: {}%'.format((round(100 * len(x_test) / colored.shape[0], 2))))

x_train = set(range(0, colored.shape[0]))
for item in x_test:
    x_train.remove(item)
x_train = list(x_train)

x_train = colored[sorted(x_train)]
x_test = colored[sorted(x_test)]

# convert from color image (RGB) to grayscale; grayscale = 0.299*red + 0.587*green + 0.114*blue
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# convert color train and test images to gray
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# make numbers floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_gray = x_train_gray.astype('float32')
x_test_gray = x_test_gray.astype('float32')

# input image dimensions
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

# reshape images to row x col x channel for CNN
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)


############### build the autoencoder model ###############
# FIRST - network parameters
input_shape = (img_rows, img_cols, 1)
batch_size = 32*4
kernel_size = 3
latent_dim = 256
layer_filters = [64, 128, 256]  # encoder/decoder number of CNN layers and filters per layer

# SECOND - build the encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
shape = K.int_shape(x)
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)  # generate a latent vector
encoder = Model(inputs, latent, name='encoder')  # instantiate encoder model
# encoder.summary()

# THIRD - build the decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
# stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)
outputs = Conv2DTranspose(filters=channels,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder')  # instantiate decoder model
# decoder.summary()

# FORTH - create autoencoder model; autoencoder = encoder + decoder
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
# autoencoder.summary()


###############  prepare model saving directory and callbacks  ###############
save_dir = os.path.join(repo_path, 'saved_ae_models')
# model_name = 'colorized_ae_model.{epoch:03d}.h5'
model_name = 'colorized_ae_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

# save weights for future use (e.g. reload parameters w/o training)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')

# early stopping if validation loss does not improve after 11 epochs
earlystopping = EarlyStopping(monitor='val_loss',
                              patience=11)

# called every epoch
callbacks = [lr_reducer, checkpoint, earlystopping]


############### training the autoencoder ###############
# load a trained model
# autoencoder = load_model(os.path.join(save_dir, 'colorized_ae_model.h5'))

autoencoder.fit(x_train_gray,
                x_train,
                validation_data=(x_test_gray, x_test),
                epochs=100,
                batch_size=batch_size,
                callbacks=callbacks)


############### get prediction and display 16 images ###############
# x_decoded = autoencoder.predict(x_test_gray)
#
# random_16 = np.random.randint(0, x_test.shape[0], size=8)  # get random 16 numbers
#
# # display og version
# imgs = x_test[random_16]
# imgs = imgs.reshape((4, 2, img_rows, img_cols, channels))
# imgs = np.vstack([np.hstack(i) for i in imgs])
# plt.figure(figsize=(8, 8))
# plt.axis('off')
# plt.title('Test color images (Ground  Truth)')
# plt.imshow(imgs, interpolation='none')
# plt.savefig('{}/test_color.png'.format(save_dir))
# # plt.show()
#
# # display grayscale version of test images
# imgs = x_test_gray[random_16]
# imgs = imgs.reshape((4, 2, img_rows, img_cols))
# imgs = np.vstack([np.hstack(i) for i in imgs])
# plt.figure(figsize=(8, 8))
# plt.axis('off')
# plt.title('Test gray images (Input)')
# plt.imshow(imgs, interpolation='none', cmap='gray')
# plt.savefig('{}/test_gray.png'.format(save_dir))
# # plt.show()
#
# # display re-colorized images
# imgs = x_decoded[random_16]
# imgs = imgs.reshape((4, 2, img_rows, img_cols, channels))
# imgs = np.vstack([np.hstack(i) for i in imgs])
# plt.figure(figsize=(8, 8))
# plt.axis('off')
# plt.title('Colorized test images (Predicted)')
# plt.imshow(imgs, interpolation='none')
# plt.savefig('{}/test_recolorized.png'.format(save_dir))
# # plt.show()

end_time = time.time()
print('{} seconds to run this python module'.format(round(end_time - start_time)))


############### get prediction of fun pictures ###############
repo_path = os.path.dirname(os.getcwd())

fun = io.imread(os.path.join(repo_path + '/fun_test_photos/ASHISH PUJARI.jpeg'))
resize_size = 2 ** 7
funny = [resize(fun, (resize_size, resize_size))]

fun = io.imread(os.path.join(repo_path + '/fun_test_photos/YURI.jpeg'))
fun = resize(fun, (resize_size, resize_size))
funny.append(np.array(fun))

fun = io.imread(os.path.join(repo_path + '/fun_test_photos/ANN.jpeg'))
fun = resize(fun, (resize_size, resize_size))
funny.append(np.array(fun))

funny = np.array(funny)

num_pics = len(funny)
funny_gray = rgb2gray(funny)
funny = funny.astype('float32')
funny_gray = funny_gray.astype('float32')
funny = funny.reshape(funny.shape[0], img_rows, img_cols, channels)
funny_gray = funny_gray.reshape(funny_gray.shape[0], img_rows, img_cols, 1)

autoencoder = load_model(os.path.join(save_dir, 'colorized_ae_model.h5'))
x_decoded = autoencoder.predict(funny_gray)

# display og version
imgs = funny
print(len(funny))
imgs = imgs.reshape((num_pics, 1, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure(figsize=(8, 8))
plt.axis('off')
# plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('{}/fun_color.png'.format(save_dir))
# plt.show()

# display grayscale version of test images
imgs = funny_gray
imgs = imgs.reshape((num_pics, 1, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure(figsize=(8, 8))
plt.axis('off')
# plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('{}/fun_gray.png'.format(save_dir))
# plt.show()

# display re-colorized images
imgs = x_decoded
imgs = imgs.reshape((num_pics, 1, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure(figsize=(8, 8))
plt.axis('off')
# plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('{}/fun_recolorized.png'.format(save_dir))
# plt.show()

print('done')
