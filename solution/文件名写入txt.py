import os
import shutil

base_path = r"D:\8022biaozhu_jiancha\188\zz_finally\test\images"
save_path = r"D:\8022biaozhu_jiancha\188\zz_finally\test.txt"

base_list = set()

for task in os.listdir(base_path):
    #label = task.split(".")[0]
    base_list.add(task)
print(len(base_list))


with open(save_path, "w",encoding="utf-8") as wf:
    for line in base_list:
        wf.write(line + "\n")
