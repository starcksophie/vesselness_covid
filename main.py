import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import nibabel as nib
from tqdm import tqdm

import analytics
import writer 

from skimage import morphology as morpho
from skimage.measure import label
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

def normalize(im):
    return 255*(im - im.mean())/im.max()

def apply_mask(image, mask):
    # return image * mask
    expanded = np.stack((mask, mask, mask), axis=3)
    return image*expanded

def main(filepath, maskpath, rorpo_out=None):
    analytics.result = {}
    img_mask = nib.load(maskpath).get_fdata()
    # segmentation
    if rorpo_out is not None:
        print("loading segmentation...")
        seg = nib.load(rorpo_out).get_fdata()
    else:
        print("loading data..")
        img = nib.load(filepath).get_fdata()
        # preprocessing
        print("normalizing...")
        for i, slice in enumerate(img):
            img[i] = normalize(slice)
        #FIXME: exec rorpo 
    # post processing
    print("applying some post processing...")
    seg = apply_mask(seg, img_mask)
    seg_2d = np.zeros((seg.shape[0:3]))

    # for i, slice in tqdm(enumerate(seg)):
    for i, s in enumerate(seg):
        gray = rgb2gray(s)
        # gray = slice
        # Otsu raises ValueError if single grayscale value.
        if len(np.unique(gray)) > 1:
            ots =  threshold_otsu(gray)
            # print(f"shape: {gray.shape}, thresh: {ots}, i = {i}\n")
            # plt.imshow(gray)
            # plt.show()
            seg_2d[i] = (gray > ots).astype(int)
            seg_2d[i] *= morpho.erosion(img_mask[i], morpho.disk(2))
        else:
            seg_2d[i] = np.zeros((gray.shape))
    
    analytics.get_analytics(seg, img_mask, verbose=True)
    distance_map = analytics.distance(seg_2d)
    analytics.label_value(distance_map)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    writer.write_result()
