import os
import cv2
import numpy as np
from tqdm import tqdm

#from chinese import change_cv2_draw

# color_list = [(220,20,60),(199,21,133),(128,0,128),(75,0,130),(0,0,255),(0,255,255),(0,100,0),(255,255,0),(255,165,0),(0,0,0),(128,128,0)]
# color_list = [(220,20,60),(199,21,133),(128,0,128),(75,0,130),(0,0,255),(128,128,0)]
color_list = [(0,0,255),(0,255,0),(0,255,255),(255,0,0),(0,0,255),(0,165,255),(0,0,0)]
# labels_list = ['body','face','head','torso','smoke','fire','tricycle','motorcycle','car','truck','bicycle']
labels_list = ['gun','qiang','two_wheel','pet','box','face','6','7','8','9','10']


def xywh2xyxy(infos, w, h):
    cx = int(float(infos[1]) * w)
    cy = int(float(infos[2]) * h)
    r_w = int(float(infos[3]) * w)
    r_h = int(float(infos[4]) * h)
    x1 = cx - r_w//2
    y1 = cy - r_h//2
    x2 = x1+r_w
    y2 = y1+r_h
    return x1,y1,x2,y2


def drow_once(label_file, img_file, check_img_file):
    with open(label_file, 'r') as lf:
        labels = lf.readlines()
        # 0-body  1-face
        # im = cv2.imread(img_file)
        # print(img_file)
        im = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)
        h,w = im.shape[:2]
        if len(labels) != 0:
            for label in labels:
                infos = label.strip(' ').strip('\n').split(" ")
                x1,y1,x2,y2 = xywh2xyxy(infos, w, h)
                # if label == "delete":
                if infos[0] == "11":
                    im[int(y1):int(y2), int(x1):int(x2)] = 114
                    continue
                #     continue
                # if infos[0] == "0" :
                # if infos[0] == '6':
                #     infos[0] = '5'
                # if infos[0] == "1" or infos[0] == "2":
                label_name = labels_list[int(infos[0])]
                color = color_list[int(infos[0])]
                # color = (255,0,0)
                cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=2)
                # cv2.putText(im,label_name,(x1, y1),0,tl / 3,(220,20,60),thickness=tf, lineType=cv2.LINE_AA)
                # im = change_cv2_draw(im, label_name, (x1, y1), sizes=35, colour=(220, 20, 60))
            # cv2.imwrite(check_img_file, im)
            cv2.imencode('.jpg', im)[1].tofile(check_img_file)

if __name__ == '__main__':
    
    # imgs_path = r'/data/Pubilc/small_algo/detection_eleven/facebody_dataset2.0/res_clean/res_source/res_2'
    # labs_path =r'/data/Pubilc/small_algo/detection_eleven/facebody_dataset2.0/org/labels'
    # save_path = r'/data/Pubilc/small_algo/detection_eleven/facebody_dataset2.0/res_clean/body_down'

    imgs_path = r"D:\8022biaozhu_jiancha\202\jyz1225"  # 图片的路径
    # labs_path =r'C:\Users\18655\Desktop\temp\res_clean\res_labels'
    labs_path =r"D:\8022biaozhu_jiancha\202\labels"  # 标签的路径
    save_path = r"D:\8022biaozhu_jiancha\202\img"    # 保存的画了框的结果图的路径

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for img in tqdm(os.listdir(imgs_path)):
        label = img.split(".")[0]+".txt"
        if label not in os.listdir(labs_path):
            continue
        label_path =  os.path.join(labs_path,label)
        img_path = os.path.join(imgs_path,img)
        save_img_path = os.path.join(save_path,img)
        drow_once(label_file=label_path,img_file=img_path,check_img_file=save_img_path)