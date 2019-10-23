from PIL import Image
import os
from skimage import io
# import cv2


def black_and_white(input_image_path, output_image_path):
    color_image = Image.open(input_image_path)
    bw = color_image.convert('L')
    bw.save(output_image_path)


# def black_and_white_opencv(input_image_path, output_image_path):
#     image = cv2.imread(input_image_path)
#     grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(output_image_path, grayImage)


repo_path = '/Users/anneitrheim/PycharmProjects/Colorize-Black-and-White-Photos/'

file_name = [f for f in os.listdir(repo_path + 'images-colored') if f.endswith(('.jpg', '.JPG'))]

for i in range(0, len(file_name)):
    black_and_white(os.path.join(repo_path + 'images-colored', file_name[i]),
                    os.path.join(repo_path + 'images-bw', file_name[i]))

print("Black and white pictures created and saved.")

# TODO when converted to b+w and it is the same, delete it... some are already b+w

# create a dictionary of the pictures (read in as arrays)
X_DICT_BW_TRAIN = dict()
Y_DICT_COLOR_TRAIN = dict()
# X_DICT_BW_TEST = dict()
# Y_DICT_COLOR_TEST = dict()

for img_id in file_name:
    bw = io.imread(os.path.join(repo_path + 'images-bw', img_id))
    color = io.imread(os.path.join(repo_path + 'images-colored', img_id))

    # train_xsz = int(3 / 4 * bw.shape[0])  # use 75% of image as train and 25% for validation

    X_DICT_BW_TRAIN[img_id] = bw
    Y_DICT_COLOR_TRAIN[img_id] = color
    # X_DICT_BW_TEST[img_id] = bw[train_xsz:, :, :]
    # Y_DICT_COLOR_TEST[img_id] = color[train_xsz:, :, :]

print("Dictionary of pictures created.")
