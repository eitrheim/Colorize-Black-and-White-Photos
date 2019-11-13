import cv2
import os
import numpy as np

def CreateMatrix(test=False):
    i = 0
    x_data = []
    if (test == False):
        repo_path = '../colored-resized/'
    else:
        repo_path = '../TestPhoto/'
    files = [f for f in os.listdir(repo_path) if f.endswith(('.jpg', '.JPG', '.tif'))]
    for myFile in files:
        try:
            print(myFile)
            image = cv2.imread(myFile)
            x_data.append(image)
            i += i
            if i % 100 == 0:
                print("%d images to array" % i)
        except ValueError:
            print("Value Error: ", myFile)
    if (test == False):
        print('Training Data Matrix Shape: ', np.array(x_data).shape)
    else:
        print("Test Data Matrix Shape: ", np.array(x_data).shape)
    return x_data