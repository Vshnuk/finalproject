import cv2
import numpy as np
from PIL import Image

def main():
    img_bgr = cv2.imread("inh.jpg")
    height, width, channel = img_bgr.shape
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    k=img.shape[0]
    l=img.shape[1]

    randomnumber={}
    encrypted = np.zeros((height, width,3), np.uint8)
    random = np.zeros((height, width,3), np.uint8)
    for i in range (k):#traverses through height of the image
        for j in range (l): #traverses through width of the image
#                original[i,j]=img[i,j]
            randomnumber[i,j]=np.random.randint(0,255)
            random[i,j]=randomnumber[i,j]
#                img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
            # print(randomnumber[i,j])
            
            # Encryption

            if(((randomnumber[i,j]>=128) and (img[i,j]>=128))):
                encrypted[i,j]=img[i,j]-128
                temp=img[i,j]-128
            elif(((randomnumber[i,j]<128) and (img[i,j]<128))):
                encrypted[i,j]=img[i,j]
                temp=img[i,j]
            elif(((randomnumber[i,j]>=128) and (img[i,j]<128))):
                encrypted[i,j]=img[i,j]+128
                temp=img[i,j]+128
            else:
                encrypted[i,j]=img[i,j]
                temp=img[i,j]

#                enc_img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
            print(encrypted[i,j])
        encrypt_image=Image.fromarray(encrypted)
        encrypt_image.save("encrypt.jpg")
if __name__ == '__main__':
    main()