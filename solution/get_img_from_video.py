import os 
import cv2 

def cut(base_path, save_path):
    videos = os.listdir(base_path)
    interval = 50
    num = 0
    for v in videos:
        print(os.path.join(base_path, v))
        cap = cv2.VideoCapture(os.path.join(base_path, v))
        count = 1
        # num = 0

        while True:
            r, f = cap.read()
            if not r:
                break

            if count % interval == 0:
                name = os.path.join(save_path, "Box_det_20231229_train" + "_{}.jpg".format(num))
                # name = os.path.join(save_path, v[:-4] + "_{}.jpg".format(num))
                # dirp = os.path.join(save_path,v[:-4])
                # if not os.path.exists(dirp):
                #     os.makedirs(dirp)
                # name = os.path.join(dirp, "pk_t40up_20230510_" + v[:-4] + "_{}.jpg".format(num))
                cv2.imwrite(name, f)
                num += 1
                
            count += 1
        cap.release()
    print(num)


if __name__ == "__main__":
    base_path = r"D:\8022biaozhu_jiancha\204\vv"
    save_path = r"D:\8022biaozhu_jiancha\204\img"
    # base_path2 = r"D:\8022biaozhu_jiancha\191\testall_testout_2023-12-04091647.951805\old"
    # save_path2 = r"D:\8022biaozhu_jiancha\191\testall_testout_2023-12-04091647.951805\old_img"
    cut(base_path, save_path)
    # cut(base_path2, save_path2)
        