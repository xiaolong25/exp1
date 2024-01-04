from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def imgpath(path):
    yolox = os.path.join(path,'new_img')   # 左
    yolov5 = os.path.join(path,'old_img')   # 右
    img_xs = os.listdir(yolox)
    pathlist = []
    for img_x in img_xs:
        # if '-' in img_x:
        #     img_x_ = img_x.split('-')[0] + '.jpg'
        # else:
        #     img_x_ = img_x[:-7] + '.jpg'
        p1 = os.path.join(yolox,img_x)
        p2 = os.path.join(yolov5,img_x)
        if not os.path.exists(p2):
            continue
        tmplist = [p1,p2,img_x]
        pathlist.append(tmplist)
    return pathlist

def two2one(img1p,img2p,save):
    w = 1920
    h = 1080
    img1 = Image.open(img1p)
    img1 = img1.resize((w, h),Image.ANTIALIAS)
    img2 = Image.open(img2p)
    img2 = img2.resize((w, h),Image.ANTIALIAS)
    result = Image.new(img1.mode, (w*2, h ))
    result.paste(img1, box=(0, 0))
    result.paste(img2, box=(w, 0))
    result.save(save)
    # plt.imshow(result)
    # plt.show()

if __name__ == "__main__":
    path = r'D:\8022biaozhu_jiancha\191\testall_testout_2023-12-04091647.951805'
    savep = r"D:\8022biaozhu_jiancha\191\testall_testout_2023-12-04091647.951805\merge1"
    pathlists = imgpath(path)
    for pathlist in tqdm(pathlists):
        path1,path2 = pathlist[0],pathlist[1]
        save = os.path.join(savep,pathlist[2])
        two2one(path1,path2,save)