from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np

def ColConvert(matrix ,direction):
    print("Coloring and Splitting Data!")
    if direction == "rbg2lab":
        for i in range(matrix.shape[0]):
            image = matrix[i]
            # image = np.transpose(image)
            if i == 0:
                X = rgb2lab(1.0 / 255 * image)[:, :, 0]
                Y = rgb2lab(1.0 / 255 * image)[:, :, 1:]
                Y = Y / 128
                X = X.reshape(1, 200, 200, 1)
                Y = Y.reshape(1, 200, 200, 2)
            else:
                tempX = rgb2lab(1.0 / 255 * image)[:, :, 0]
                tempY = rgb2lab(1.0 / 255 * image)[:, :, 1:]
                tempY = tempY / 128
                tempX = tempX.reshape(1, 200, 200, 1)
                tempY = tempY.reshape(1, 200, 200, 2)
                X = np.vstack([X, tempX])
                Y = np.vstack([Y, tempY])
            if i % 100 == 0 and i > 0:
                print("%d images colorized" % i)
    print("%d images colorized" % i)
    print("Done!")
    return X, Y