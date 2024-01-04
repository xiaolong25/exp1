import os
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch

def cos_sim(tensor_list):
    data = ''
    all_im = []
    for x in tqdm(tensor_list):  # x:[[py],[wk]]
    # for i, result in enumerate(py_res):
        tmp = [0,0]
        py_res = torch.tensor(x[0])
        wk_res = torch.tensor(x[1])
        feat_res = torch.nn.functional.normalize(torch.tensor(py_res.flatten()), p=2, dim=0)
        feat_caffe_res = torch.nn.functional.normalize(torch.tensor(wk_res.flatten()), p=2, dim=0)
        dot_nom_result = np.dot(feat_res, feat_caffe_res)  # 向量点积或矩阵乘法

        # check if result are same by cosine distance
        dot_result = np.dot(py_res.flatten(), wk_res.flatten())
        left_norm = np.sqrt(np.square(py_res).sum())  
        right_norm = np.sqrt(np.square(wk_res).sum())
        cos_sim = dot_result / (left_norm * right_norm)

        # print("cos sim between onnx and caffe models: {}  nom:{}".format(cos_sim, dot_nom_result))
        tmp[0] = cos_sim.tolist()
        tmp[1] = float(dot_nom_result)
        # print(type(tmp[0]), type(tmp[1]))
        # print(type(cos_sim.tolist()), type(dot_nom_result))
        data += str(cos_sim.tolist()) + " " + str(float(dot_nom_result)) + '\n'
        all_im.append(tmp)
        # print(tmp)
    data = 'cos_sim' + "                     " + 'dot_nom_result' + '\n' + data

    all_0 = 0
    all_1 = 0
    for i in all_im:
        all_0 += i[0]
        all_1 += i[1]
    # print(all_0)
    # print(all_1)
    
    avg_cos_sim = all_0 / len(all_im)
    avg_dot_nom_result = all_1 / len(all_im)
    print("*****************")
    print(f"余弦距离：{avg_cos_sim}")
    
    data += "\n\n\n" + "avg_cos_sim: " + str(avg_cos_sim) + "  " + "avg_dot_nom_result: " + str(avg_dot_nom_result)
    # with open(cos_sim_txt,'w')as wf:
    #     wf.write(data)
    #     wf.close()

def consistence(txt_py, txt_wk):
    py_list = []
    wk_list = []
    with open(txt_py,'r')as rf:
        labels = rf.readlines()
        for lab in labels:
            lab = lab.strip("\n").split(" ")
            lab_1 = []
            for x in lab:
                if x != '':
                    lab_1.append(x)
            py_list.append(lab_1)

    with open(txt_wk,'r')as rf:
        labels = rf.readlines()
        for lab in labels:
            lab = lab.strip("\n").split(" ")
            lab_2 = []
            for x in lab:
                if x != '':
                    lab_2.append(x)
            # print(lab)
            wk_list.append(lab_2)


    m = 0
    n = 0
    y = 0
    err_conf_py = 0.0
    err_conf_wk = 0.0

    ttt = ''
    for a in py_list:
        for b in wk_list:
            if a[0] == b[0]:
                y += 1
                # print(y)
                # print(a[2])
                # print(b[2])
                # if a[1] == "1_yes" and b[1] == "1_yes" and a[2] == "0_no" and b[2] == "1_yes":
                if a[2][0] != b[2]:
                    m += 1
                    ttt += a[0] + '\n'

    print(f"图片总数:{y}")
    print(f"分类不一致图片数: {m}")
    print(f"分类一致率：{(y-m)/len(py_list)}")
    # print(f"avg_err_conf_py: {err_conf_py/(m + 0.00001)}")
    # print(f"avg_err_conf_wk: {err_conf_wk/(m + 0.00001)}")
    # print(f"err_conf_diff: {err_conf_py/(m + 0.00001)-err_conf_wk/(m + 0.00001)}")
    print("*****************")

    # with open (r"C:\Users\jixia\Desktop\txt.txt","w")as wf:
    #     wf.write(ttt)

if __name__ == '__main__':
    py_res = r"C:\Users\jixia\Desktop\py_res.txt"
    wk_res = r"C:\Users\jixia\Desktop\c++res.txt"
    # cos_sim_txt = r"C:\Users\jixia\Desktop\cos_res.txt"
    py_list = []
    wk_list = []
    with open(py_res,'r')as rf:
        labels = rf.readlines()
        for lab in labels:
            lab = lab.strip("\n").split(" ")
            lab_1 = []
            for x in lab:
                if x != '':
                    lab_1.append(x)
            py_list.append(lab_1)
    with open(wk_res,'r')as rf:
        labels = rf.readlines()
        for lab in labels:
            lab = lab.strip("\n").split(" ")
            lab_2 = []
            for x in lab:
                if x != '':
                    lab_2.append(x)
            wk_list.append(lab_2)

    tensor_list = []
    for a in py_list:
        for b in wk_list:
            if a[0] == b[0]:
                # print(a)
                # print(b)
                tensor12 = [[float(a[3]),float(a[4])],
                            [float(b[3]),float(b[4])]]
                tensor_list.append(tensor12)
    cos_sim(tensor_list)

    consistence(py_res, wk_res)

            
    