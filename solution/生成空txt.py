import os
import tqdm

path1 = r"D:\8022biaozhu_jiancha\200\all_data\images"
path2 = r"D:\8022biaozhu_jiancha\200\all_data\labels"

list1 = os.listdir(path1)
list2 = os.listdir(path2)

x = 0
for img in tqdm.tqdm(list1):
    img = img.split('.')[0] + '.txt'
    if img not in list2:
        file = os.path.join(path2,img)
        with open(file, "w") as lf:
            lf.close
        x += 1
print(x)
