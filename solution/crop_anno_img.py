from genericpath import exists
import os
import os.path as path
from random import shuffle
import shutil
import cv2
import argparse

from tqdm import tqdm

#扣画的框的小图

# parser = argparse.ArgumentParser(description='crop annotation img')
# parser.add_argument('--anno_path', type=str, default=r'D:\detect2_res\detect\labels',
#                     help='annotation path')
# parser.add_argument('--img_path', type=str, default=r'D:\detect2_res\detect\images',
#                     help='img path')
# parser.add_argument('--res_path', type=str, default=r'D:\detect2_res\work',
#                     help='result path')
# args = parser.parse_args()

# dir_map = {'0':'body','1':'face','2':'head','3':'torso','4':'smoke','5':'fire','6':'tricycle','7':'motorcycle','8':'car','9':'truck','10':'bicycle','11':'delete'}
dir_map = {'0':'Bottle cap','1':'Can','2':'Cup','3':'Juice box','4':'Packaging','5':'Plastic bag','6':'Plastic bottle','7':'Plastic','8':'Undefined trash','9':'Wood'}
def get_anno(shape, file):
    f = open(file, 'r')
    annos = []
    lines = f.readlines()
    for line in lines:
        obj = line.split(' ')
        # if obj[0] in dir_map.keys():
        cx = float(obj[1])
        cy = float(obj[2])
        w = float(obj[3])
        h = float(obj[4])
        xmin = int(shape[1] * (cx - w/2))
        ymin = int(shape[0] * (cy - h/2))
        xmax = int(shape[1] * (cx + w/2))
        ymax = int(shape[0] * (cy + h/2))
        annos.append((obj[0], xmin, ymin, xmax, ymax))
    return annos

def crop_annos(img, annos, file_name):
    idx = 1
    for anno in annos:
        sub_img = img[anno[2]:anno[4], anno[1]:anno[3]]
        fname = file_name+'_'+'{0:02d}'.format(idx) + ".jpg"
        # tmp = dir_map[anno[0]] + fname
        label_class = anno[0]
        tmp1 = os.path.join(res_path, dir_map[label_class])
        tmp = os.path.join(tmp1,fname)
        # print(tmp)
        # if not os.path.exists(tmp1):
        #     os.mkdir(tmp1)
        try:
            cv2.imwrite(tmp, sub_img)
        except:
            print(tmp)
        # print('{} / {}'.format(anno[0], fname))
        idx += 1

""" 
import ptvsd
ptvsd.enable_attach(address = ('192.168.1.113', 5678))
ptvsd.wait_for_attach() 
 """

if __name__ == '__main__':
    # args.anno_path += '\\'the file not exist: /data/Public/small_algo/detection_eleven/dataset_2.0/huizong/imagessmokefire20220527_9999.jpg
    # args.img_path += '\\'/data/Public/small_algo/detection_eleven/dataset_2.0/huizong/images/classes.jpg
    # args.res_path += '\\'Traceback (most recent call last):
    
    #本地
    anno_path = r"D:\8022biaozhu_jiancha\188\08072023.v1i.yolov5pytorch_can\train\labels\\"
    res_path = r"D:\8022biaozhu_jiancha\188\08072023.v1i.yolov5pytorch_can\train\xiaotu\\"
    img_path = r"D:\8022biaozhu_jiancha\188\08072023.v1i.yolov5pytorch_can\train\images\\"

    #服务器
    # anno_path = r'/data/Pubilc/small_algo/detection_eleven/facebody_dataset2.0/org/labels/'   #标签路径
    # res_path = r'/home/wujiale/anno/'                                                         #保存小图路径
    # img_path = r'/data/Pubilc/small_algo/detection_eleven/facebody_dataset2.0/org/images/'    #图片路径

    for k in dir_map.keys():
        sub_dir = res_path + dir_map[k]
        dir_map[k] = sub_dir
        if not path.exists(sub_dir):
            os.mkdir(sub_dir)
    print(dir_map)
    
    # if os.path.exists(res_path):
    #     shutil.rmtree(res_path)
    # os.mkdir(res_path)

    

    for _, _, files in os.walk(img_path):
        for file in tqdm(files):
            # print(file)
            img_file = img_path + file
            anno_file = anno_path + file[0:-4] + '.txt'
            if not path.exists(anno_file):
                print(img_file)
                print('the file not exist:', anno_file)
                # img_file = img_file.replace('.jpg', '.png')
                # if not path.exists(img_file):
                #     print('the file not exist:', img_file)
                continue

            img = cv2.imread(img_file)
            annos = get_anno(img.shape, anno_file)
            crop_annos(img, annos, file[0:-4])
            
            #print('file:{} anno sum:{}'.format(file, len(annos)))