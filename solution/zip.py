import os
import zipfile
import shutil
from pathlib import Path
from multiprocessing import Process

base_path = r"H:\dataset_face"


#压缩文件
def zip(zips, s, e):
    save_path = r"H:\dataset_face_res"
   
    for sub_path in zips[s:e]:
        sp = os.path.join(base_path, sub_path)
        dp = os.path.join(save_path, sub_path+".zip")

        zip = zipfile.ZipFile(dp,"w",zipfile.ZIP_DEFLATED)
        for dirpath, dirnames, filenames in os.walk(sp):
            fpath = dirpath.replace(sp,'') #这一句很重要，不replace的话，就从根目录开始复制
            fpath = fpath and fpath + os.sep or ''#这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
            for filename in filenames:
                zip.write(os.path.join(dirpath, filename),fpath+filename)
        zip.close()

#解压文件
def unzip(zips, s, e):
    save_path = r"H:\dataset_face_res"

    for z in zips[s:e]:
        if not z.endswith(".zip") and not z.endswith(".rar"):
            continue
        z = Path(z)
        file = os.path.join(base_path, z.name)
        d_file = os.path.join(save_path, z.stem)
        try:
            os.mkdir(d_file)
        except:
            print(f"文件夹已存在：{z}")
        pwd = "4spbbix6s9DvWXt"
        cmd_str = "7z x {} -o{} -p{}".format(file, d_file,pwd)
        # print(f"{cmd_str}")
        os.system(cmd_str)


if __name__ == "__main__":
    # num_check1()
    # print("\n\n")
    # num_check2()
   
    
    zipss = os.listdir(base_path)
    zips = []
    #unzip(zips, 0, 4)
    for task in zipss:
        flog = task.split("_")[0]
        if flog == "0" or flog == "1" or flog == "3" or flog == "4" or flog == "5" or flog == "2" or task.split(".")[0] == "6_0" or task.split(".")[0] == "7_1" or task.split(".")[0] == "7_2" or task.split(".")[0] == "8_5":
            continue
        zips.append(task)
    # print(f"{len(zips)}")

    process_list = []
    for i in range(4):
        tmp = Process(target=unzip, args=(zips, i * 6, i * 6 + 6))
        process_list.append(tmp)
        tmp.start()
    
    for p in process_list:
        p.join()
    
