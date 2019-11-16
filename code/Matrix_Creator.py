import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

def CreateMatrix(test=False):
    print("Reshaping images and converting to matrix!")
    i = 0
    image_height = 200
    image_width = 200
    channels = 3
    if test == False:
        repo_path = '../colored-resized/'
    else:
        repo_path = '../TestPhoto/'
    files = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]
    x_data = np.ndarray(shape=(len(files),image_height, image_width, channels),
                         dtype=np.float32)
    for myFile in files:
        try:
            img = load_img(repo_path + myFile)
            x = img_to_array(img)
            x_data[i] = x
            i += 1
            if i % 100 == 0:
                print("%d images to array" % i)
        except ValueError:
            print("Value Error: ", myFile)
    print("%d images to array" % i)
    print("Import Complete!")
    if test == False:
        print('Training Data Matrix Shape: ', np.array(x_data).shape)
    else:
        print("Test Data Matrix Shape: ", np.array(x_data).shape)
    print("Done!")
    return x_data