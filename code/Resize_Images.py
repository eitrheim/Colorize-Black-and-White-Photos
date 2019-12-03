import cv2
import os

width = 200
height = 200
import_path = "../Images/"
export_path = "../colored-resized/"

file_name = [f for f in os.listdir(import_path) if f.endswith(('.jpg', '.JPG', '.tif'))]

for _file in file_name:
    img = cv2.imread(import_path + _file, cv2.IMREAD_UNCHANGED)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    file = export_path + _file[:-4] + ".jpg"
    cv2.imwrite(file, resized)

cv2.waitKey(0)
cv2.destroyAllWindows()