import os
import cv2
import numpy as np


def Max_point(list,list1):
    list1 = []
    max_w = 0
    max_h = 0
    min_w = 1000
    min_h = 1000
    for i in range(len(list)):
        if i%2 == 0:
            if list[i] >max_w:
                max_w = list[i]
            if list[i]< min_w:
                min_w = list[i]
        else:
            if list[i] >max_h:
                max_h = list[i]
            if list[i]< min_h:
                min_h = list[i]
    list1.append(max_w)
    list1.append(min_w)
    list1.append(max_h)
    list1.append(min_h)
    return list1
if __name__ == '__main__':
    img_path = r"C:\Users\jixia\Desktop\no_background\box_package"
    txt_save = r"C:\Users\jixia\Desktop\222222\txt_save"
    save_image = r"C:\Users\jixia\Desktop\222222\save_image"
    resize_image = r"C:\Users\jixia\Desktop\222222\save_r"
    imgs = os.listdir(img_path)
    i =1
    point = np.array([[1039, 3], [1039, 289], [1895, 3], [1895, 755]])
    for image in imgs:
        img = cv2.imread(img_path+"/"+image)
        img_d = img.copy()
        img_r = img.copy()
        w = int(img_r.shape[1] * 0.05 if img_r.shape[1] * 0.05 > 5 else 5)
        h = int(img_r.shape[0] * 0.05 if img_r.shape[0] * 0.05 > 5 else 5)
        w += img_r.shape[1]
        h += img_r.shape[0]

        img_r = cv2.resize(img_r, (0, 0),fx=0.5, fy=0.5)
        img = img_r

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d_value = 0 if img[0][0] < 10 else 255
        
        rh,rw,_=img_r.shape
        cv2.line(img_r, (0,0), (0,rh), (d_value,d_value,d_value), 3)  # 划线
        cv2.line(img_r, (0,0), (rw,0), (d_value,d_value,d_value), 3)
        cv2.line(img_r, (rw,rh), (0,rh), (d_value,d_value,d_value), 3)
        cv2.line(img_r, (rw,rh), (rw,0), (d_value,d_value,d_value), 3)  # 画了个正方形
        cv2.imwrite(os.path.join(save_image, image), img_r)

        # cv2.imwrite(os.path.join(txt_save,image.split(".")[0]+"_1.jpg"), img)
        # cv2.imshow("img",img)
        # 二值化
        thr_value = 10 if img[0][0] < 10 else 240
        ret,thresh=cv2.threshold(img, thr_value, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(os.path.join(txt_save,image.split(".")[0]+"_2.jpg"), thresh)
        # cv2.imshow("thresh",thresh)
        binary,contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#得到轮廓信息  函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性
        j = 0
        size = 1
        for bin in range(len(binary)):
            jump = False
            for p in binary[bin]:
                if p[0][0] ==0 and p[0][1] == 0:
                    jump = True
                    break

            if jump:
                continue

            if len(binary[bin])>size:
                size = len(binary[bin])
                j = bin
            
            #情况一，传入list，元素是numpy数组，函数将会画出相邻两点之间的连线
        # imgnew = cv2.drawContours(img, binary, 1, (0,255,0), 2)#第三个参数是1，代表了取出contours中index为1的array
        # cv2.imshow("imgnew",imgnew)
            #情况二，传入numpy数组，函数将仅仅会画出这些点，而不会将他们依次连接
        # x1= Max_point(binary[j].reshape(-1, 1),[])[0]+3
        # x2= max(1,Max_point(binary[j].reshape(-1, 1),[])[1]-3)
        # y1= Max_point(binary[j].reshape(-1, 1),[])[2]+3
        # y2= max(1,Max_point(binary[j].reshape(-1, 1),[])[3]-3)
        
        #cv2.imwrite(os.path.join(save_image,image.split(".")[0]+".jpg"), img1[int(y2):int(y1),int(x2):int(x1)])
        # img_d = cv2.drawContours(img_d, binary[j], -1, (0,0,255), 1)#第三个参数为-1，代表了画出contours[0]中所有的点
        # # cv2.imshow("imgnew",imgnew)
        # cv2.imwrite(os.path.join(save_image,image.split(".")[0]+"_o.jpg"), img_d)
        img_r = cv2.drawContours(img_r, binary[j], -1, (0,0,255), 3)#第三个参数为-1，代表了画出contours[0]中所有的点
        # # cv2.imshow("imgnew",imgnew)
        cv2.imwrite(os.path.join(resize_image,image.split(".")[0]+"_r.jpg"), img_r)
        # open(os.path.join(txt_save,image.split(".")[0]+".txt"), 'w')
        # cv2.fillPoly(img1,[binary[j]],(144,144,144))
        # cv2.imwrite(os.path.join(txt_save,image.split(".")[0]+"_3.jpg"), img1)
        np.savetxt(os.path.join(txt_save,image.split(".")[0]+".txt"),binary[j].reshape(-1, 1),fmt="%d",newline=' ')
        
