#!/usr/bin/env python
# coding: utf-8

# the images for this are available at 
# https://drive.google.com/open?id=1_gxVdFLl5jPFb4uba2Tccj4Tdq6qvkR7
# The additional 60,000 images from wiki are here:
# https://drive.google.com/open?id=1iaR9oGS-rPbake-lCkD43OkM8YmlWCCx

# Sample code for this CNN was taken from 
# https://github.com/nikhitmago/deep-cnn-for-image-colorization/blob/master/(Deep)%20CNNs%20for%20Image%20Colorization.ipynb

# In[84]:


#loading images as a list 
from PIL import Image
from numpy import asarray
from os import listdir
from matplotlib import image
import os
from PIL import Image
import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dropout, MaxPooling2D, Conv2D, Dense, Flatten,BatchNormalization
from keras import optimizers
from skimage.color import rgb2grey
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Set up image path

# In[62]:


repo_path = '/Users/antonovaval89/Desktop/UofChicagoInfo/Deep_Learning/Final_Project/Images/'
file_name = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]


# In[63]:


file_name[0]


# In[64]:


image_bw = io.imread(repo_path + 'images-bw/'+ file_name[45])


# In[65]:


image_bw[0][199]


# ### Create Uniformly Sized Color Images

# In[303]:


def image_resize(repopath, filename):
    for i in range(0, len(file_name)):
        image = io.imread(repo_path + file_name[i])
        if (len(image.shape)<3): # to account for grayscale images possibly in dataset
            continue             # will skip over a grayscale image
        image2 = Image.fromarray(image)
        img_resized = image2.resize((200,200))
        img_resized.save(repo_path +'images-colored/' + file_name[i])


# In[304]:


image_resize(repo_path, file_name)


# In[1]:


#wiki data set has some images which are black and white but import as a 3-dim matrix. 
#is there any way to avoid this?
#image = io.imread('/Users/antonovaval89/Desktop/UofChicagoInfo/Deep_Learning/Final_Project/00/86700_1958-07-28_1980.jpg')

#print(image.shape)
#print(image)
#io.imshow(image)


# In[3]:


#import numpy as np
#import skimage
#from skimage import data, io, filters, color, exposure
#import matplotlib.pyplot as plt
#%matplotlib inline

#for col, channel in zip('rgb', np.rollaxis(image, axis=-1)):
#    hist, bin_centers = exposure.histogram(channel)
#    plt.fill_between(bin_centers, hist, color=col, alpha=0.3)


# In[4]:


#def show_images(images,titles=None):
#    """Display a list of images"""
#    n_ims = len(images)
 #   if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
  #  fig = plt.figure()
  #  n = 1
  #  for image,title in zip(images,titles):
  #      a = fig.add_subplot(1,n_ims,n) # Make subplot
  #      if image.ndim == 2: # Is image grayscale?
  #          plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
   #     plt.imshow(image)
#        a.set_title(title)
#        n += 1
#    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
#    plt.show()


# In[4]:


#red, green, blue = image.copy(), image.copy(), image.copy()
#red[:,:,(1,2)] = 0
#green[:,:,(0,2)] = 0
#blue[:,:,(0,1)] = 0

#show_images(images=[red, green, blue], titles=['Red Intensity', 'Green Intensity', 'Blue Intensity'])
#print ('Note: lighter areas correspond to higher intensities\n')


# ### Creating Black and White Images:

# In[305]:


def black_and_white(input_image_path, output_image_path):
    color_image = Image.open(input_image_path)
    bw = color_image.convert('L')
    bw.save(output_image_path)


# In[306]:


for i in range(0, len(file_name)):
    black_and_white(os.path.join(repo_path + 'images-colored', file_name[i]),
                    os.path.join(repo_path + 'images-bw', file_name[i]))


# ### Uploading Images as a list

# In[247]:


repo_path = '/Users/antonovaval89/Desktop/UofChicagoInfo/Deep_Learning/Final_Project/Images'
file_name = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]


# In[248]:


# load all images in a directory
from os import listdir
from matplotlib import image
# load all images in a directory
bw_images = list()
for name in file_name:
# load image
    bw_data = image.imread(repo_path + '/images-bw/'+ name)
# store loaded image
    bw_images.append(bw_data)


# In[249]:


color_images = list()
for name in file_name:
# load image
    color_data = image.imread(repo_path + '/images-colored/'+ name)
# store loaded image
    color_images.append(color_data)


# In[250]:


plt.imshow(bw_images[1300], cmap='gray')


# In[251]:


plt.imshow(color_images[1300])


# ### Converting Image Lists to Arrays

# In[260]:


bw = np.array(bw_images).astype('float32')/255
color = np.array(color_images).astype('float32')/255


# In[271]:


bw = bw.reshape(1301, 200, 200,1)
color = color.reshape(1301, 200, 200,3)


# In[272]:


print('Black and White Shape',bw.shape)
print('Color Shape',color.shape)


# In[264]:


(200/1301)*100


# In[273]:


X_train = bw[0:-200]
X_test = bw[-200:]

y_train = color[0:-200]
y_test = color[-200:]


# cnn = Sequential()
# cnn.add(Conv2D(128, kernel_size = (5,5), strides=(1, 1), 
#                padding='same', input_shape = (200,200,1)))
# #cnn.add(BatchNormalization())
# cnn.add(MaxPooling2D(pool_size=(2, 2), 
#                      strides=(1, 1), padding='same'))
# cnn.add(Conv2D(128, kernel_size = (5,5), 
#                strides=(1, 1), padding='same'))
# #cnn.add(BatchNormalization())
# cnn.add(MaxPooling2D(pool_size=(2, 2), 
#                      strides=(1, 1), padding='same'))
# #cnn.add(BatchNormalization())
# #cnn.add(Flatten())
# #cnn.add(Dropout(0.25))
# cnn.add(Dense(69,input_shape=(3,), activation='relu'))
# #cnn.add(BatchNormalization())
# #cnn.add(Dropout(0.5))
# cnn.add(Dense(3,input_shape=(3,), activation='softmax'))
# cnn.summary()

# In[325]:


cnn = Sequential()
cnn.add(Conv2D(64, kernel_size = (10,10), strides=(1, 1), 
               padding='same', input_shape = (200,200,1)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(128, kernel_size = (5,5), strides=(1, 1), 
               padding='same', input_shape = (200,200,1)))
cnn.add(MaxPooling2D(pool_size=(2, 2), 
                     strides=(1, 1), padding='same'))
cnn.add(Conv2D(128, kernel_size = (5,5), 
               strides=(1, 1), padding='same'))
cnn.add(BatchNormalization())
#cnn.add(Flatten())
cnn.add(Dropout(0.25))
cnn.add(Dense(64,input_shape=(3,), activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))
cnn.add(Dense(3,input_shape=(3,), activation='softmax'))
cnn.summary()


# In[ ]:


cnn.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy', 'mse'])
history = cnn.fit(X_train, y_train, epochs=50, validation_split=0.15)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In[ ]:


predictions = cnn.predict(X_test)


# In[ ]:


predictions[0].shape


# In[ ]:


predictions[0].dtype


# In[ ]:


predictions[0]*=255


# In[ ]:


y_test[0]


# In[ ]:


plt.imshow(y_test[0])


# In[ ]:


plt.imshow(predictions[0].astype('uint8'))


# In[ ]:





# In[ ]:




