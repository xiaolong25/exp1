import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np

p1 = r"C:\Users\jixia\Desktop\i"
p2 = r"C:\Users\jixia\Desktop\images"
for na in tqdm(os.listdir(p1)):
    pp = os.path.join(p1,na)
    im = cv2.imread(pp)
    # im = cv2.imdecode(np.fromfile(pp, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print("***")
    img = cv2.resize(im, (640, 384), interpolation=cv2.INTER_LINEAR)
    # na = na[:-4] + '.bmp'
    cv2.imwrite(os.path.join(p2,na),img)