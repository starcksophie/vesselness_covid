import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import nibabel as nib
from ipywidgets import interactive, fixed
from tqdm import tqdm

from skimage import morphology as morpho
from skimage.measure import label
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

def normalize(im):
    return 255*(im - im.mean())/im.max()

def apply_mask(image, mask):
    for i, slice in tqdm(enumerate(image)):
        for j in range(3):
            image[i][:, :, j] *= mask[i]
    return image

def main(filepath, maskpath, rorpo_out=None):
    # print("loading data..")
    # img = nib.load(filepath).get_fdata()
    img_mask = nib.load(maskpath).get_fdata()
    # # preprocessing
    # print("normalizing...")
    # for i, slice in enumerate(img):
    #     img[i] = normalize(slice)
    # segmentation
    if rorpo_out is not None:
        print("loading segmentation...")
        seg = nib.load(rorpo_out).get_fdata()
    else:
        pass
        # exec rorpo
    # post processing
    print("applying some post processing...")
    seg = apply_mask(seg, img_mask)
    for i, slice in tqdm(enumerate(seg)):
        gray = slice@[0.3, 0.6, 0.1]
        seg[i] = (gray > threshold_otsu(gray)).astype(int)
    # compute distance map
    print("computing distance map...")
    distance_map = seg
    skeleton = distance_map
    for i, slice in enumerate(distance_map):
        distance_map[i] = ndi.distance_transform_edt(seg[i])
        skeleton[i] = skeletonize(seg[i])
    plt.imshow(distance_map[185])



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])