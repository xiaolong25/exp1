import cv2
import numpy as np
import os
from tqdm import tqdm

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    ssim_im = (ssim_map*255).astype(np.uint8)
    return ssim_map.mean(), ssim_im

def ssim2(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    # print(ssim_map.mean())
    ssim_im = (ssim_map*255).astype(np.uint8)
    ch, cw = ssim_im.shape
    cssim_im = cv2.cvtColor(ssim_im, cv2.COLOR_GRAY2BGR)
    h, w = img1.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    c[0:ch, 0:cw] = cssim_im
    return ssim_map.mean()
    # return c

def ssim3(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    print(window)
    mu1 = cv2.filter2D(img1, -1, window)  # valid
    print(mu1)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    print(ssim_map.mean())                                                    
    ssim_im = (ssim_map*255).astype(np.uint8)
    ch, cw = ssim_im.shape
    cssim_im = cv2.cvtColor(ssim_im, cv2.COLOR_GRAY2BGR)
    h, w = img1.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    c[0:ch, 0:cw] = cssim_im
    return c




root_dir = r"C:\Users\jixia\Desktop\ceshi\res_file"
target_dir = r"C:\Users\jixia\Desktop\ceshi\res_file"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for sub_dir in os.listdir(root_dir)[:1000]:
    if sub_dir.find(".DS") > -1: continue
    sub_path = os.path.join(root_dir, sub_dir)
    # print(sub_path)
    files = os.listdir(sub_path)
    im_path1 = os.path.join(sub_path, files[0])
    im_path2 = os.path.join(sub_path, files[1])
    im1g = cv2.imread(im_path1, 0)
    im2g = cv2.imread(im_path2, 0)
    im1 = cv2.imread(im_path1)
    #print(im_path1)
    im2 = cv2.imread(im_path2)
    if im1.shape != im2.shape: continue
    im1g = cv2.resize(im1g, (16,16))
    im2g = cv2.resize(im2g, (16,16))

    cssim_im = ssim2(im1g, im2g)
    break
    # v = cssim_im.mean()/255.
    #
    # #print(im1.shape, im2.shape, cssim_im.shape)
    # im = np.concatenate((im1,im2, cssim_im),axis = 0)
    # target_path = os.path.join(target_dir, sub_dir+"_%.2f.jpg"%v)
    # cv2.imwrite(target_path, im)

