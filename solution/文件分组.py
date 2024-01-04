import shutil
import os
import random

path_1_img = r"D:\8022biaozhu_jiancha\night_dataset\images"
path_1_lab = r"D:\8022biaozhu_jiancha\night_dataset\labels"
path_2 = r"D:\8022biaozhu_jiancha\night_dataset\tasks"

list = os.listdir(path_1_img)
random.shuffle(list)            #将一个列表中的元素打乱顺序
img_num = len(list) # 总的图片数
dir_num = 18  # 分多少个task
dir_img_num = img_num/dir_num # 单个文件夹数量
print("平均：{}".format(dir_img_num))
i=1 #总数
num = 1 #文件夹总数

while i <= dir_num:
    j=0 #分的数量
    name = path_2 + "\\task" + "_" + str(num)
    os.makedirs(name)
    imgname = name + "\\images"
    labname = name + "\\labels"
    os.makedirs(imgname)
    os.makedirs(labname)
    for img in list:
        if j < dir_img_num:
            j += 1
            imgpath = os.path.join(path_1_img, img)
            labpath = os.path.join(path_1_lab, (img.split(".")[0] + ".txt"))
            saveimgpath = os.path.join(imgname, img)
            savelabpath = os.path.join(labname, (img.split(".")[0] + ".txt"))
            shutil.copy(imgpath, saveimgpath)
            shutil.copy(labpath, savelabpath)
            list.remove(img)
        if j >= dir_img_num:
            break
    print(j)
    print("task_{}分组完成！".format(num))
    num += 1
    i += 1