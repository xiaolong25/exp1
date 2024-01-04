import os
import tqdm
import cv2
import numpy as np
# from utils import CONF

def count(path):
    img_num = 0
    lab_num = 0
    empty_img_num = 0
    # print("***********************************start*************************************")
    img_path = os.path.join(path,'images')
    if not os.path.exists(img_path):
        print("请检查该人员images文件夹名称是否正确!!!")
        return 0
    img_num = len(os.listdir(img_path))
    lab_path = os.path.join(path,'labels')
    if not os.path.exists(lab_path):
        print("请检查该人员labels文件夹名称是否正确!!!")
        return 0

    imgnas = os.listdir(img_path)
    data = ""
    des = path.split('\\')[-1]
    for imgna in tqdm.tqdm(imgnas,ncols=60,unit='img',leave=False,desc=des):
        st = ""
        # labelna = imgna.split('.')[0] + '.txt'
        labelna = imgna[:-4] + '.txt'
        if imgna[-4:] == '.txt':
            continue
        labelp = os.path.join(lab_path,labelna)
        imagep = os.path.join(img_path,imgna)
        if not os.path.exists(labelp):
            print(f"未找到文件：{labelp},请检查。")
            continue
        if os.path.getsize(labelp) == 0:
            empty_img_num += 1
            # continue
        with open(labelp,'r') as rf:
            cur_num = 0
            labels = rf.readlines()
            # im = cv2.imread(imagep)
            im = cv2.imdecode(np.fromfile(imagep, dtype=np.uint8), cv2.IMREAD_COLOR)
            h,w = im.shape[:2]
            boxlist = []
            for label in labels:
                if len(label) == 0:
                    print(f"{labelp} is err !!!")
                    continue
                infos = label.strip(' ').strip('\n').split(" ")
                x1,y1,x2,y2 = xywh2xyxy(infos, w, h)
                boxlist.append([x1,y1,x2,y2])
            org_boxlist = boxlist.copy()
            for box in org_boxlist:
                if len(boxlist) == 1:
                    lab_num += 1
                    cur_num += 1
                else:
                    boxlist.remove(box)
                    num = 0
                    for compar in boxlist:
                        thr = IoU(box,compar)
                        if thr <= iou_thr:   
                            num += 1
                    if num == len(boxlist):
                        lab_num += 1
                        cur_num += 1
        st = imgna + " ---> " + str(cur_num)
        data += st + "\n"
    return empty_img_num,data,img_num,lab_num

def IoU(box1, box2) -> float:
    weight = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    s_inter = weight * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union

def xywh2xyxy(infos, w, h):
    cx = int(float(infos[1]) * w)
    cy = int(float(infos[2]) * h)
    r_w = int(float(infos[3]) * w)
    r_h = int(float(infos[4]) * h)
    x1 = cx - r_w//2
    y1 = cy - r_h//2
    x2 = x1+r_w
    y2 = y1+r_h
    return x1,y1,x2,y2

def conut_rerson(na,path_task):
    box_num = 0
    data_all = ''
    txtna = "res_" + path_task.split('\\')[-1] + ".txt"
    name = na
    txtpath = os.path.join(path_task,'data')
    if not os.path.exists(txtpath):
        os.makedirs(txtpath)
        print(f"{path_task}中无data文件夹, 已经自动创建。")
    # txt = os.path.join(txtpath,txtna)  # 打印txt在data下
    txt = os.path.join(path_all,txtna)
    empty_img_num,data,img_num,lab_num = count(path_task)
    # title = "img_name ---> box_num" + "\n"
    title = "图片名 ---> 图中框数量" + "\n"
    # data = title + data + '\n'  + 'img_num = ' + str(img_num) + '\n' + 'empty_img_num = ' + str(empty_img_num) + '\n' + 'box_num = ' + str(box_num) + '\n' + 'reward = ' + str(reward)
    # with open(txt ,'w') as wf:
    #     wf.write(data)
    #     wf.close()
    reward = price*lab_num
    # print(f"人员{name}标注图片数量：{img_num}张，总的框数量：{lab_num}个!!!")
    print("\033[1;32m------>Person:\033[0m",end='')
    print("\033[1;35m%s\033[0m"%name,end='')
    print("\033[1;32m, image_num:%d, empty_img_num:%d, box_num:%d, reward:%.2f\033[0m"%(img_num,empty_img_num,lab_num,reward))
    # print("\033[1;32m------>Person:%s, image_num:%d, empty_img_num:%d, box_num:%d, reward:%.2f\033[0m"%(name,img_num,empty_img_num,lab_num,reward))
    # print("************************************end**************************************")
    box_num += lab_num
    data_all += na + "---->" + str(img_num) + "----------->" + str(empty_img_num) + "----------------->" + str(lab_num) + "--------->" + str(reward)
    # data = title + data + '\n'  + 'img_num = ' + str(img_num) + '\n' + 'empty_img_num = ' + str(empty_img_num) + '\n' + 'box_num = ' + str(box_num) + '\n' + 'reward = ' + str(reward)
    data = title + data + '\n'  + '图片数量 = ' + str(img_num) + '\n' + '无框图片数量 = ' + str(empty_img_num) + '\n' + '框数量 = ' + str(box_num) + '\n' + 'rmb = ' + str(reward)
    with open(txt ,'w') as wf:
        wf.write(data)
        wf.close()
    return empty_img_num,data_all,box_num,reward,img_num

def finddirs(path):
    if os.path.isdir(path):
        dirs = os.listdir(path)
        if 'images' in dirs and 'labels' in dirs:
            return path
        else:
            for dir in dirs:
                dirpath = os.path.join(path,dir)
                paths = finddirs(dirpath)
                if paths is not None:
                    return paths
    else:
        return

def print_res(flg,all_img_num,all_empty_img_num,box_num,all_reward):
    if flg == 0:
        print("\033[1;32m**************************************__Counting Start__**************************************\033[0m")
    if flg == 1:
        print("\033[1;35m+--------------------------+\033[0m")
        print("\033[1;35m|\033[0m",end = '')
        print("\033[1;35mResult:\033[0m")
        print("\033[1;35m|\033[0m",end = '')
        print(f"\033[1;32m---> all_img_num: {all_img_num}\033[0m")
        print("\033[1;35m|\033[0m",end = '')
        print(f"\033[1;32m---> all_empty_img_num: {all_empty_img_num}\033[0m")
        print("\033[1;35m|\033[0m",end = '')
        print(f"\033[1;32m---> all_box_num: {box_num}\033[0m")
        print("\033[1;35m|\033[0m",end = '')
        print("\033[1;32m---> all_reward: %.2f\033[0m"%all_reward)
        print("\033[1;35m|\033[0m",end = '')
        print(f"\033[1;32m---> price: {price}\033[0m")
        print("\033[1;35m+--------------------------+\033[0m")
        print("\033[1;32m*********************************************End*********************************************\033[0m")

def conut_all(path_all):
    box_num = 0
    all_reward = 0
    all_img_num = 0
    all_empty_img_num = 0
    print_res(0,all_img_num,all_empty_img_num,box_num,all_reward)
    dirs = os.listdir(path_all)
    data_all = ''
    if 'images' not in dirs or 'labels' not in dirs: 
        for dir in dirs:
            dir_path = os.path.join(path_all,dir)
            name = dir_path.split("\\")[-1]
            dir_path_ = finddirs(dir_path)
            if dir_path_ is None:
                continue
            empty_img_num,data, cur_box_nums, reward, img_num= conut_rerson(name,dir_path_)
            all_reward += reward
            box_num += cur_box_nums
            data_all += data + "\n"
            all_img_num += img_num
            all_empty_img_num += empty_img_num
    else:
        name = path_all.split("\\")[-1]
        empty_img_num, data, cur_box_nums, reward, img_num= conut_rerson(name,path_all)
        all_reward += reward
        box_num += cur_box_nums
        data_all += data + "\n"
        all_img_num += img_num
        all_empty_img_num += empty_img_num
    res_file = os.path.join(path_all,'res_all.txt')
    # if os.path.exists(res_file):
    #     shutil.rmtree(res_file)


    # data_all =  'name     img_num      empty_img_num      box_num     reward' + '\n' + data_all + '\n'
    # data_all += 'all_img_num = ' + str(all_img_num) + '\n'
    # data_all += 'all_empty_img_num = ' + str(all_empty_img_num) + '\n'
    # data_all += 'all_box_num = ' + str(box_num) + '\n'
    # data_all += 'all_reward = ' + str(all_reward) 

    data_all =  '姓名     图片数量      无框图片数量      框数量     rmb' + '\n' + data_all + '\n'
    data_all += '总的图数量 = ' + str(all_img_num) + '\n'
    data_all += '总的无框图的数量 = ' + str(all_empty_img_num) + '\n'
    data_all += '总的框数 = ' + str(box_num) + '\n'
    data_all += '总的金额 = ' + str(all_reward) 
    with open(res_file,'w')as w:
        w.write(data_all)
        w.close()
    print_res(1,all_img_num,all_empty_img_num,box_num,all_reward)

if __name__ == "__main__":
#############################################################################################################
    iou_thr = 0.92                                                # <---- iou阈值控制框数量，谨慎修改!!! ---->
    price = 0.05                                                  # 每个框的单价
    path_all = r'D:\数据标注\230727包裹标注'                       # 任务路径,个人/总的
    conut_all(path_all)                                           # 统计
#############################################################################################################
 