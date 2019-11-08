from keras.preprocessing.image import img_to_array, load_img
import os
import numpy as np
from PIL import Image


#Creating folder of images
repo_path = '../images-colored/'
file_name = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]

print("Working with {0} images".format(len(file_name)))


train_files = []
y_train = []
i=0
for _file in file_name:
    train_files.append(_file)
    label_in_file = _file.find("_")
    y_train.append(_file[0:label_in_file])

print("Files in train_files: %d" % len(train_files))

# Original Dimensions
image_width = 200
image_height = 200

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), channels, image_height, image_width),
                     dtype=np.float32)

i = 0
for _file in train_files:
    img = load_img(repo_path + _file)  # this is a PIL image
    try:
        # Convert to Numpy Array
        x = img_to_array(img)
        x = x.reshape((3, image_width, image_height))
        # Normalize
        x = (x - 128.0) / 128.0
        dataset[i] = x
        i += 1
        if i % 100 == 0:
            print("%d images to array" % i)
    except ValueError:
        print("Value Error: ", _file)
print("All images to array!")


