import os
import cv2
from PIL import Image

def findpixel():
   pass





if __name__ == "__main__":
    path = "./"
    save_path = "./"
    images = os.listdir(path)
    for im in images:
        image = cv2.imread(os.path.join(path,im,),1)
        w,h = image.shape()
