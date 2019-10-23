from PIL import Image
import os
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

file_name = [f for f in os.listdir(repo_path + 'data') if f.endswith(('.jpg', '.JPG'))]

for i in range(0, len(file_name)):
    black_and_white(os.path.join(repo_path + 'data', file_name[i]),
                    os.path.join(repo_path + 'images', 'bw_'+file_name[i]))

# TODO when converted to b+w and it is the same, delete it... some are already b+w
