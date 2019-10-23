from PIL import Image
import os


def black_and_white(input_image_path, output_image_path):
    color_image = Image.open(input_image_path)
    bw = color_image.convert('L')
    bw.save(output_image_path)


repo_path = '/Users/anneitrheim/PycharmProjects/Colorize-Black-and-White-Photos/'
file_name = [f for f in os.listdir(repo_path + 'data') if f.endswith('.jpeg')]

for i in range(0, len(file_name)):
    black_and_white(os.path.join(repo_path + 'data', file_name[i]),
                    os.path.join(repo_path + 'images', 'bw_'+file_name[i]))
