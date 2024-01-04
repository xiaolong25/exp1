# -*- coding: utf-8 -*-

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
import clip
import copy
from PIL import Image
import argparse
import numpy as np
import os
import shutil
import random
from html4vision import Col, imagetable

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".JPG", ".JPEG", ".WEBP", ".BMP", ".PNG"]

def make_parser():
    parser = argparse.ArgumentParser("kmeans tools")
    parser.add_argument("--K", default=10, type=int, help="num clusters")
    parser.add_argument("--batch_size", default=512, type=int, help="batch size")
    parser.add_argument("--num_worker", default=4, type=int, help="num_worker")
    parser.add_argument("--max_epoch", default=100, type=int, help="max_epoch")
    parser.add_argument("--image_dir", default=None, type=str, help="image_dir")
    parser.add_argument("--save_image_dir", default=None, type=str, help="save_image_dir")
    return parser

def kmeans(K, image_features):
    n_img = image_features.shape[0]
    device = image_features.device
    center_index = np.random.choice(np.arange(n_img), size=K, replace=False)
    center_index = torch.Tensor(center_index).to(device).to(torch.int64)  # [K, ]
    centers = image_features[center_index].clone()  # [K, 512]
    groups = [[] for _ in range(K)]
    groups_index = [[] for _ in range(K)]
    end = False
    epoch_id = 0
    while not end:
        for i in range(n_img):
            image_feature = image_features[i].clone()
            min_d = (centers[0] - image_feature).pow(2.0).sum().sqrt()
            min_kindex = 0
            for k in range(1, K, 1):
                dist = (centers[k] - image_feature).pow(2.0).sum().sqrt()
                if dist < min_d:
                    min_d = dist
                    min_kindex = k
            groups[min_kindex].append(image_feature.unsqueeze(0))
            groups_index[min_kindex].append(i)
        # update centers
        for k in range(K):
            groups[k] = torch.cat(groups[k], 0)
            groups[k] = torch.mean(groups[k], dim=0, keepdim=True)
        new_centers = torch.cat(groups, 0)
        # center_dist = (centers - new_centers).pow(2.0).sum().sqrt()
        center_dist = (centers - new_centers).pow(2.0).sum(1).sqrt().mean()
        center_dist = float(center_dist.cpu().detach().numpy())
        epoch_id += 1
        progress_str = "epoch: %d, center_dist: %.6f" % (epoch_id, center_dist)
        print(progress_str)
        if center_dist < 0.00001:
            end = True
        if not end:
            centers = new_centers
            groups = [[] for _ in range(K)]
            groups_index = [[] for _ in range(K)]
    return groups_index

def kmeans_fast(K, image_features, batch_size, max_epoch):
    n_img = image_features.shape[0]
    device = image_features.device
    img_indexes = torch.arange(n_img, dtype=torch.int64, device=device)
    center_index = np.random.choice(np.arange(n_img), size=K, replace=False)
    center_index = torch.Tensor(center_index).to(device).to(torch.int64)  # [K, ]
    centers = image_features[center_index].clone()  # [K, 512]
    groups_init = [[] for _ in range(K)]
    groups = copy.deepcopy(groups_init)
    groups_index = copy.deepcopy(groups_init)
    end = False
    epoch_id = 0
    steps = n_img // batch_size
    if n_img % batch_size != 0:
        steps += 1
    while not end:
        group_ids = []
        for step_id in range(steps):
            start_i = step_id * batch_size
            end_i = (step_id + 1) * batch_size
            if end_i > n_img:
                end_i = n_img
            image_features_ = image_features[start_i:end_i, :].unsqueeze(1)  # [batch_size, 1, 512]
            centers_ = centers.unsqueeze(0)  # [1, K, 512]
            distants = (image_features_ - centers_).pow(2.0).sum(-1).sqrt()  # [batch_size, K]
            group_id = torch.argmin(distants, 1, keepdim=False)  # [batch_size, ]  # 每张图片被分到的簇id
            group_ids.append(group_id)
        group_ids = torch.cat(group_ids, 0)  # [n_img, ]  # 每张图片被分到的簇id
        # update centers
        for k in range(K):
            groups_index[k] = img_indexes[group_ids == k]
            if len(groups_index[k]) == 0:  # 很奇怪，为什么有的簇没有图片
                groups[k] = centers[k:k+1, :].clone()
                groups_index[k] = center_index[k:k+1].clone()
            else:
                groups[k] = image_features[group_ids == k]
                groups[k] = torch.mean(groups[k], dim=0, keepdim=True)
        new_centers = torch.cat(groups, 0)
        center_dist = (centers - new_centers).pow(2.0).sum(1).sqrt().mean()
        center_dist = float(center_dist.cpu().detach().numpy())
        epoch_id += 1
        progress_str = "epoch: [%d/%d], center_dist: %.6f" % (epoch_id, max_epoch, center_dist)
        print(progress_str)
        if epoch_id >= max_epoch:
            end = True
        if not end:
            centers = new_centers
            groups = copy.deepcopy(groups_init)
            groups_index = copy.deepcopy(groups_init)
    return groups_index



class DirDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_dir="",
                 preprocess=None):
        self.image_dir = image_dir
        image_names_dirty = os.listdir(image_dir)
        self.image_names = []
        for image_name in image_names_dirty:
            ext = os.path.splitext(image_name)[1]
            if ext in IMAGE_EXT:
                self.image_names.append(image_name)
        self.n_img = len(self.image_names)
        self.preprocess = preprocess

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img = Image.open(os.path.join(self.image_dir, image_name))
        tensor = self.preprocess(img)
        return image_name, tensor



'''
export CUDA_VISIBLE_DEVICES=0

nohup python clip_kmeans.py --K 4000 --image_dir ../chache/20230814/images --save_image_dir ../chache_20230814     > clip_kmeans.log 2>&1 &


nohup python clip_kmeans.py --K 10 --image_dir ../crop_bbox/0 --save_image_dir ../crop_bbox_new     > clip_kmeans.log 2>&1 &


rm -rf kmeans.zip; zip -r kmeans.zip ../chache_20230814

'''
if __name__ == "__main__":
    args = make_parser().parse_args()
    K = args.K

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # build Dataset and DataLoader
    dataset = DirDataset(image_dir=args.image_dir, preprocess=preprocess)
    image_names = copy.deepcopy(dataset.image_names)
    n_img = len(dataset)
    assert K < n_img
    assert K > 1
    sampler = SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             num_workers=args.num_worker,
                                             drop_last=False,
                                             sampler=sampler)
    num_steps = len(dataloader)

    image_features = []
    # print_interval = max(num_steps // 5, 1) + 1
    print_interval = 20
    for i, (image_name, image) in enumerate(dataloader):
        image = image.to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image)  # [N, 512]
            image_feature /= image_feature.norm(dim=-1, keepdim=True)  # [N, 512]  归一化
            image_features.append(image_feature)
        if i % print_interval == 0:
            progress_str = "get image_feature: [{}/{}]".format(i + 1, num_steps)
            print(progress_str)
    image_features = torch.cat(image_features, 0)  # [n_img, 512]
    print("Start k-means clustering algorithm...")
    with torch.no_grad():
        # groups_index = kmeans(K, image_features)
        groups_index = kmeans_fast(K, image_features, args.batch_size, args.max_epoch)
    # save result
    if os.path.exists(args.save_image_dir):
        shutil.rmtree(args.save_image_dir)
    os.makedirs(args.save_image_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_image_dir, 'keep'), exist_ok=True)
    for k in range(K):
        os.makedirs(os.path.join(args.save_image_dir, "%d"%k), exist_ok=True)
        for i in groups_index[k]:
            shutil.copy(os.path.join(args.image_dir, image_names[i]), os.path.join(args.save_image_dir, "%d"%k, image_names[i]))
        # 每簇随机取1个
        index = random.randint(0, len(groups_index[k]) - 1)
        i = groups_index[k][index]
        shutil.copy(os.path.join(args.image_dir, image_names[i]), os.path.join(args.save_image_dir, "keep", image_names[i]))
    print('Save result in \'%s/keep\'' % args.save_image_dir)
    cols = [
        Col('id1', 'ID'),  # make a column of 1-based indices
    ]
    for k in range(K):
        cols.append(Col('img', '%d'%k, '%s/%d/*.jpg'%(args.save_image_dir, k)))
    imagetable(cols)




