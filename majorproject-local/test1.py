import cv2
import numpy as np
from PIL import Image

def main():
    img_bgr = cv2.imread("encrypt.jpg")
    height, width, channel = img_bgr.shape
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    k=img.shape[0]
    l=img.shape[1]

    for i in range (k):#traverses through height of the image
        for j in range (l): #traverses through width of the image
            print(img_bgr[i,j])

if __name__ == '__main__':
    main()