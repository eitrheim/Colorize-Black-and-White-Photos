#!/usr/bin/env python
# coding: utf-8

# the images for this are available at 
# https://drive.google.com/open?id=1_gxVdFLl5jPFb4uba2Tccj4Tdq6qvkR7
# The additional 60,000 images from wiki are here:
# https://drive.google.com/open?id=1iaR9oGS-rPbake-lCkD43OkM8YmlWCCx

# In[301]:


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


# ### Set up image path

# In[302]:


repo_path = '/Users/antonovaval89/Desktop/UofChicagoInfo/Deep_Learning/Final_Project/Images/'
file_name = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]


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


# In[310]:


#wiki data set has some images which are black and white but import as a 3-dim matrix. 
#is there any way to avoid this?
#image = io.imread('/Users/antonovaval89/Desktop/UofChicagoInfo/Deep_Learning/Final_Project/00/86700_1958-07-28_1980.jpg')
#io.imshow(image)
#print(image.shape)


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


# ### Creating a dictionary of BW and Color Images

# In[113]:


bw_dict= dict()
color_dict = dict()
for img_id in file_name:
    bw = io.imread(os.path.join(repo_path + '/images-bw', img_id))
    color = io.imread(os.path.join(repo_path + '/images-colored', img_id))
    bw_dict[img_id] = bw
    color_dict[img_id] = color


# In[195]:


image_names = list(bw_dict)


# In[165]:


pyplot.imshow(bw_dict[image_names[0]], cmap='gray')


# ### Uploading Images as a list

# In[ ]:


repo_path = '/Users/antonovaval89/Desktop/UofChicagoInfo/Deep_Learning/Final_Project/Images'
file_name = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]


# In[223]:


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


# In[224]:


color_images = list()
for name in file_name:
# load image
    color_data = image.imread(repo_path + '/images-colored/'+ name)
# store loaded image
    color_images.append(bw_data)


# In[ ]:




