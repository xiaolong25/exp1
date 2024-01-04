import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
import shutil
from multiprocessing import Process
import argparse
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def cal_ssim(im, ims, device):
    im_data1 = np.transpose(im, (2, 0, 1))
    size = len(ims)
    ims = np.array(ims)
    #print('----1----')
    im_data2 = np.transpose(ims, (0, 3, 1, 2))
    im_X = torch.from_numpy(im_data1).unsqueeze(0).float().to(device).repeat(size,1,1,1)
    im_Y = torch.from_numpy(im_data2).float().to(device)
    #print(im_X.shape, im_Y.shape)

    ssim_val = ssim(im_X, im_Y, data_range=255, size_average=False)
    #print('----2----')

    s = np.max(ssim_val.cpu().numpy())
    return s

# def cal_ssim2(idx, device_ims):
#     s_im = device_ims[idx].unsqueeze(0).repeat(device_ims.shape[0], 1, 1, 1)
#     ssim_val = ssim(im_X, im_Y, data_range=255, size_average=False)


def get_diverse_img(img_list, root_dir, target_dir, device, task_id):
    ims = []
    for im_name in img_list:
        im = cv2.imread(os.path.join(root_dir, im_name))
        im = cv2.resize(im, (48, 48))
        ims.append(im)
    #print("total ims %d"%len(ims))
    target_ims = [ims[0]]
    target_files = [img_list[0]]
    for idx in np.arange(1, len(ims)):
        if task_id == 0:
            print("----------->%d|%d|%d"%(idx, len(ims), len(target_ims)))
        s_im = ims[idx]
        svalue = cal_ssim(s_im, target_ims, device)
        if svalue < 0.8:
            target_ims.append(s_im)
            target_files.append(img_list[idx])
    for f in target_files:
        shutil.copy(os.path.join(root_dir, f), target_dir)

def get_prefix(name):
    name_c = name
    if name.startswith("@@"):
        name = name[2:]
    elif name.startswith("@"):
        name = name[1:]
    prefix = name.split("@f")[0]
    #print(name, prefix)
    return prefix

def get_img_groups(img_list):
    img_list_groups = {}
    for im_file in tqdm(img_list):
        prefix = get_prefix(im_file)
        if prefix in img_list_groups.keys():
            img_list_groups[prefix].append(im_file)
        else:
            img_list_groups[prefix] = [im_file]

    return img_list_groups

def get_chunks(img_list, task_num):
    img_list_group = get_img_groups(img_list)
    img_list_group = sorted(img_list_group.values(), key=lambda x:len(x))
    #print(img_list_group)
    for i in range(len(img_list_group)):
        print(len(img_list_group[i]))
    #for x  in img_list_group[-1]:
        #print(x)
    #print(img_list_group[-1])

    split_img_lists = [[] for _ in range(task_num)]
    pbar = tqdm(total=len(img_list_group))
    for idx in range(len(img_list_group)):
        img_list = img_list_group[idx]
        split_img_lists[idx%task_num].append(img_list)
        pbar.update(1)
    return split_img_lists


def img_list_group_diverse(img_list_groups, src_dir, target_dir, device, task_id):
    from tqdm import tqdm
    if task_id == 0: 
        pbar = tqdm(img_list_groups)
    else:
        pbar = img_list_groups
    for img_list in pbar:
        get_diverse_img(img_list, src_dir, target_dir, device, task_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_list_file', type=str, default='/home/licheng/data/SmokingData/V8.0.0/total/no.txt', help='')
    parser.add_argument('--root_dir', type=str, default='/home/licheng/data/SmokingData/V8.0.0/total/no', help='')
    parser.add_argument('--target_dir', type=str, default='/home/licheng/data/SmokingData/V8.0.0/total/no_target', help='')
    parser.add_argument('--task_num', type=int, default=80, help='')
    parser.add_argument('--device_count', type=int, default=10, help='')
    opt = parser.parse_args()

    if not os.path.exists(opt.target_dir):
        os.makedirs(opt.target_dir)
    print("task count %d"%opt.task_num)
    plist = []
    with open(opt.img_list_file, 'r') as f:
        lines = f.readlines()
    img_list = [l.strip() for l in lines]
    print('--------------split task---------------------')
    chunk_img_lists = get_chunks(img_list, opt.task_num)
    print('--------------begin task---------------------')
    for task_id in range(opt.task_num):
        device = torch.device("cuda:%d"%(task_id%opt.device_count))
        p = Process(target=img_list_group_diverse, args=(chunk_img_lists[task_id], opt.root_dir, opt.target_dir, device, task_id))
        p.daemon=True
        p.start()
        plist.append(p)
    for p in plist:
        p.join()
        


