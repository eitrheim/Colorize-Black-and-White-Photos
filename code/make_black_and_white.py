from PIL import Image
import numpy as np
import os

file_name = [f for f in os.listdir('data') if f.endswith('.jpeg')]
input_path = []
output_path = []
for i in range(0, len(file_name)):
    input_path.append(os.path.join('data/', file_name[i]))
    output_path.append(os.path.join('images/', file_name[i]))


def black_and_white(input_image_path, output_image_path):
    color_image = Image.open(input_image_path)
    bw = color_image.convert('L')
    bw.save(output_image_path)


for i in range(0, len(file_name)):
    black_and_white(input_path[i], output_path[i])

