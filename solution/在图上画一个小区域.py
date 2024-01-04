import os
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
from tqdm import tqdm

def draw(path,savep):
    name = path.split('\\')[-1]
    im = Image.open(path)
    draw = ImageDraw.Draw(im)
    arialFont =ImageFont.truetype("simsun.ttc", 32, encoding="unic")
    arialFont1 =ImageFont.truetype("simsun.ttc", 25, encoding="unic")
    txt1 = '人:'
    txt2 = '站  蹲'
    txt3 = '人标签，小于以上大小时，不用画!!!'
    draw.text((10, 110), text=txt1, fill='red'  , font=arialFont)
    draw.text((60, 110), text=txt2, fill='red'  , font=arialFont)
    draw.text((10, 230), text=txt3, fill='black'  , font=arialFont1)
    draw.rectangle((69, 150, 84, 220), fill='blue')
    draw.rectangle((124, 150, 154, 190), fill='blue')
    draw.rectangle([0,100,430,260], outline=(0,0,0))
    # im.show()
    save = os.path.join(savep,name)
    im.save(save)

if __name__ == "__main__":
    imgf = r"D:\8022biaozhu_jiancha\all\val\val"
    savep = r"D:\8022biaozhu_jiancha\all\val\save"
    imgs = os.listdir(imgf)
    for img in tqdm(imgs):
        path = os.path.join(imgf,img)
        draw(path,savep)
    