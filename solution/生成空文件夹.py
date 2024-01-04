import os
path = r'D:\8022biaozhu_jiancha\190\train\val\all'

for i in range(1,31):
    dir = os.path.join(path,str(i))
    os.makedirs(dir)