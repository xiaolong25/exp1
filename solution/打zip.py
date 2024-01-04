
import os
import zipfile
 
def getZipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
 
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()
    print("文件夹\"{0}\"已压缩为\"{1}\".".format(dirpath, outFullName))
 
 
if __name__ == "__main__":
    getZipDir(dirpath=r"D:\8022biaozhu_jiancha\36\web\images",
              outFullName=r"D:\8022biaozhu_jiancha\36\web\images\images.zip")