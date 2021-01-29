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

def apply_mask(image, mask):
    """
    image: numpy.ndarray : segmented array.
    mask: numpy.ndarray : mask array of the associated img.

    return: numpy.ndarray: the masked image.
    """
    return image * mask

def binarize(seg, img_mask):
    """
    Binarizes an image.
    seg: numpy.ndarray : segmented array.
    img_mask: numpy.ndarray : mask array of the associated img.

    return: numpy.ndarray : binarized image. 
    """
    seg_2d = np.zeros((seg.shape[0:3]))
    for i, s in enumerate(seg):
        gray = rgb2gray(s)
        # Otsu raises ValueError if single grayscale value.
        if len(np.unique(gray)) > 1:
            ots =  threshold_otsu(gray)
            seg_2d[i] = (gray > ots).astype(int)
            # remove borders
            seg_2d[i] *= morpho.erosion(img_mask[i], morpho.disk(2))
        else:
            seg_2d[i] = np.zeros((gray.shape))
    return seg_2d

def main(filepath, maskpath):
    """
    This program computes relevant metrics on intralung vessels. 

    filepath: string: path to segmentated lungs with rorpo (nifti format)
    maskpath: string: path to mask to these lungs

    """
    analytics.result = {}
    img_mask = nib.load(maskpath).get_fdata()
    print("loading\n", flush=True)
    # segmentation
    print("loading segmentation...\n", flush=True)
    seg = nib.load(filepath).get_fdata()
    # post processing
    print("applying some post processing...\n", flush=True)
    seg = apply_mask(seg, img_mask)
    seg_2d = binarize(seg, img_mask)
    print("End of slice processing\n", flush=True) 
    distance_map, skel = analytics.distance(seg_2d)
    print("distance\n", flush=True)
    dist_per_label , skel= analytics.label_value(distance_map, skel)
    print("label_value\n", flush=True) 
    analytics.get_analytics(seg, img_mask, dist_per_label, skel, verbose=True)
    print("got analytics\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    print("Did everything but write the json\n", flush=True)
    writer.write_result(sys.argv[3])
