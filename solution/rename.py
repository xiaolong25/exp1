import os
import shutil

base_name = r"D:\8022biaozhu_jiancha\127\train_jxl\0_ch"
save_name = r"D:\8022biaozhu_jiancha\127\train_jxl\0_no"

files = os.listdir(base_name)
i = 0
for file in files:
    s_f = os.path.join(base_name, file)
    d_f = os.path.join(save_name, "Ebike_nega_20230823_" + str(i) + '.jpg')

    shutil.copyfile(s_f, d_f)
    i += 1
    # txt = d_f.replace('.jpg', '.txt')
    # with open(txt, 'w') as f:
    #     pass
print(i)
