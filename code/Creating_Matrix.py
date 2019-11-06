from keras.preprocessing.image import img_to_array, load_img
import os, sys
from PIL import Image
from IPython.display import display
from IPython.display import Image as _Imgdis
import numpy as np



#Creating folder of images
folder = "../images-colored"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(onlyfiles)))


train_files = []
y_train = []
i = 0
for _file in onlyfiles:
    train_files.append("../Images" + _file)
    label_in_file = _file.find("_")
    y_train.append(_file[0:label_in_file])

print("Files in train_files: %d" % len(train_files))

# Original Dimensions
image_width = 640
image_height = 480
ratio = 4

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), channels, image_height, image_width),
                     dtype=np.float32)

i = 0
for _file in train_files:
    img = load_img(folder + "/" + _file)  # this is a PIL image
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = img_to_array(img)
    x = x.reshape((3, 120, 160))
    # Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All images to array!")

