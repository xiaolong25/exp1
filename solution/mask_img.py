from cProfile import label
from email.quoprimime import body_check
import os
import shutil
import cv2
import numpy as np
from chinese import change_cv2_draw

img_path = r"D:\8022biaozhu_jiancha\106\data\images"
label_path = r"D:\8022biaozhu_jiancha\106\data\labels"
dst_path = r"D:\8022biaozhu_jiancha\106\data\draw"
unmask_flg = False
#lab_list = ['body','face','head','torso','smoke','fire','tricycle','motorcycle','car','truck','bicycle','delete',]
#str0 = lab_list[0]

if __name__ == "__main__":
    imgs = os.listdir(img_path)
    for img in imgs:
        l_f = os.path.join(label_path, img.split(".")[0] + ".txt")
        i_f = os.path.join(img_path, img)
        image = cv2.imread(i_f)
        h,w = image.shape[:2]

        with open(l_f, 'r') as lf:
            labs = np.array([x.split() for x in lf.read().splitlines()], dtype=np.float32)
            x = labs.copy()
            if len(x) == 0:
                cv2.imwrite(os.path.join(dst_path,img),image)
                continue

            labs[:, 1] = w * (x[:, 1] - x[:, 3] / 2) # pad width
            labs[:, 2] = h * (x[:, 2] - x[:, 4] / 2) # pad height
            labs[:, 3] = w * (x[:, 1] + x[:, 3] / 2)
            labs[:, 4] = h * (x[:, 2] + x[:, 4] / 2)
            for lab in labs:
                if unmask_flg:
                    if int(lab[0]) == 0:
                        cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (0,255,255),2)
                        #str0 = lab_list[0]
                    # elif int(lab[0]) == 1:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (0,0,255),2)
                    #     #str0 = lab_list[1]
                    # elif int(lab[0]) == 2:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (255,0,0),2)
                    #     #str0 = lab_list[2]
                    # elif int(lab[0]) == 3:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (0,255,0),2)
                    #     #str0 = lab_list[3]
                    # elif int(lab[0]) == 4:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (255,255,0),2)
                    #     #str0 = lab_list[4]
                    # elif int(lab[0]) == 5:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (255,0,255),2)
                    #     #str0 = lab_list[5]
                    # elif int(lab[0]) == 6:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (125,0,125),2)
                    #     #str0 = lab_list[6]
                    # elif int(lab[0]) == 7:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (125,125,0),2)
                    #     #str0 = lab_list[7]
                    # elif int(lab[0]) == 8:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (0,125,125),2)
                    #     #str0 = lab_list[8]
                    # elif int(lab[0]) == 9:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (125,0,0),2)
                    #     #str0 = lab_list[9]
                    # elif int(lab[0]) == 10:
                    #     cv2.rectangle(image, (int(lab[1]),int(lab[2])), (int(lab[3]),int(lab[4])), (0,125,0),2)
                    #     #str0 = lab_list[10]
                    else:
                        print(f"{img}  {lab}")
                else:
                    if int(lab[0]) == 6:
                        image[int(lab[2]):int(lab[4]), int(lab[1]):int(lab[3])] = 114
        #image = change_cv2_draw(image, str0, (labs[:, 1], labs[:, 2]), sizes=15, colour=(220, 20, 60))
        cv2.imwrite(os.path.join(dst_path,img),image)