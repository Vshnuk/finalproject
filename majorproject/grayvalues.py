import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import glob



def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    
    '''

     64 | 128 |   1

    ----------------

     32 |   0 |   2

    ----------------

     16 |   8 |   4    

    '''    

    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
        if(x==0 and y==0):
            print(val)
    return val    


def main():
    images=glob.glob("uploads\\*.jpg")
    for image_file in images:
        # image_file = 'newvispic.jpg'
        img_bgr = cv2.imread(image_file)
        height, width, channel = img_bgr.shape
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        k=img.shape[0]
        l=img.shape[1]

        # original = np.zeros((height, width,3), np.uint8)
        encrypted = np.zeros((height, width,3), np.uint8)
        decrypted = np.zeros((height, width,3), np.uint8)
        # img_lbp = np.zeros((height, width,3), np.uint8)
        # enc_img_lbp = np.zeros((height, width,3), np.uint8)

        randomnumber={}
        for i in range (k):#traverses through height of the image
            for j in range (l): #traverses through width of the image
                original[i,j]=img[i,j]
                randomnumber[i,j]=np.random.randint(0,255)
                img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
                
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

                enc_img_lbp[i, j] = lbp_calculated_pixel(img, i, j)
                
                # DECRYPTION    

                if(((randomnumber[i,j]<128) and (temp<128))):
                    decrypted[i,j]=temp
                elif(((randomnumber[i,j]>=128) and (temp<128))):
                    decrypted[i,j]=temp+128
                elif(((randomnumber[i,j]>=128) and (temp>=128))):
                    decrypted[i,j]=temp-128
                elif(((randomnumber[i,j]<128) and (temp>=128))):
                    decrypted[i,j]=temp
        
        #saving encrypted image
        encrypt_image=Image.fromarray(encrypted)
        encrypt_image.save("encryptedimages\\"+image_file.split('\\')[1])
        encrypt_image=Image.fromarray(decrypted)
        encrypt_image.save(image_file.split('\\')[1])

    # caluclate histogram and plot it

    # hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    # enc_hist_lbp = cv2.calcHist([enc_img_lbp], [0], None, [256], [0, 256])

    # output_list = []
    # output_list.append({
    #     "img": original,
    #     "xlabel": "",
    #     "ylabel": "",
    #     "xtick": [],
    #     "ytick": [],
    #     "title": "Original Image",
    #     "type": "gray"        
    # })
    # output_list.append({
    #     "img": encrypted,
    #     "xlabel": "",
    #     "ylabel": "",
    #     "xtick": [],
    #     "ytick": [],
    #     "title": "Encrypted Image",
    #     "type": "gray"  
    # })
    # output_list.append({
    #     "img": decrypted,
    #     "xlabel": "",
    #     "ylabel": "",
    #     "xtick": [],
    #     "ytick": [],
    #     "title": "Decrypted Image",
    #     "type": "gray"  
    # })
    # output_list.append({
    #     "img": img_lbp,
    #     "xlabel": "",
    #     "ylabel": "",
    #     "xtick": [],
    #     "ytick": [],
    #     "title": "LBP Image",
    #     "type": "gray"
    # })
    # output_list.append({
    #     "img": enc_img_lbp,
    #     "xlabel": "",
    #     "ylabel": "",
    #     "xtick": [],
    #     "ytick": [],
    #     "title": "Encrypted LBP Image",
    #     "type": "gray"
    # })
    # output_list.append({
    #     "img": hist_lbp,
    #     "xlabel": "Bins",
    #     "ylabel": "Number of pixels",
    #     "xtick": None,
    #     "ytick": None,
    #     "title": "Histogram(LBP)",
    #     "type": "histogram"
    # })
    # output_list.append({
    #     "img": enc_hist_lbp,
    #     "xlabel": "Bins",
    #     "ylabel": "Number of pixels",
    #     "xtick": None,
    #     "ytick": None,
    #     "title": "EncryptedHistogram(LBP)",
    #     "type": "histogram"
    # })

    # output_list_len = len(output_list)
    # figure = plt.figure()
    # for i in range(output_list_len):
    #     current_dict = output_list[i]
    #     current_img = current_dict["img"]
    #     current_xlabel = current_dict["xlabel"]
    #     current_ylabel = current_dict["ylabel"]
    #     current_xtick = current_dict["xtick"]
    #     current_ytick = current_dict["ytick"]
    #     current_title = current_dict["title"]
    #     current_type = current_dict["type"]
    #     current_plot = figure.add_subplot(2, output_list_len-3,i+1)
    #     if current_type == "gray":
    #         current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
    #         current_plot.set_title(current_title)
    #         current_plot.set_xticks(current_xtick)
    #         current_plot.set_yticks(current_ytick)
    #         current_plot.set_xlabel(current_xlabel)
    #         current_plot.set_ylabel(current_ylabel)
    #     elif current_type == "histogram":
    #         current_plot.plot(current_img, color = "black")
    #         current_plot.set_xlim([0,260])
    #         current_plot.set_title(current_title)
    #         current_plot.set_xlabel(current_xlabel)
    #         current_plot.set_ylabel(current_ylabel)            
    #         ytick_list = [int(i) for i in current_plot.get_yticks()]
    #         current_plot.set_yticklabels(ytick_list,rotation = 90)
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




