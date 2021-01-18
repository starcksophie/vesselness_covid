import numpy as np
from copy import deepcopy
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize

global result

def get_analytics(img, mask, verbose=False):
    numvox = np.count_nonzero(img)
    result["voxel_nbr"] = numvox
    result["lung_pixels"] = np.count_nonzero(mask)
    if verbose:
        print('Number of voxel containing vessles: ', numvox)

def distance(seg, verbose=False):
    # compute distance map
    if verbose:
        print("computing distance map...")
    distance_map = seg
    skeleton = deepcopy(distance_map)
    for i, slice in enumerate(distance_map):
        distance_map[i] = ndi.distance_transform_edt(seg[i])
        #skeleton[i] = skeletonize(seg[i])
    if verbose:
        plt.imshow(distance_map[185])
    return distance_map, skeleton


def label_value(dist):
    label_map = label(dist)
    label_map = label_map.flatten()
    flat_dist = dist.flatten()
    label_nbr = max(label_map)
    dist_per_label = np.array([np.array([
        flat_dist[i]  for i in range(len(flat_dist)) if label_map[i] == n ])
                        for n in range(1, label_nbr)])
    mean_ = [a.mean() for a in dist_per_label]
    max_ = [a.max() for a in dist_per_label]
    min_ = [a.min() for a in dist_per_label]
    result["mean_mean_all_vessel"] = mean_.mean()
    result["std_deviation"] = np.std(mean_)
    result["max_max_all_vessel"] = max_.max()
    result["mean_max_all_vessel"] = max_.mean()
    result["mean_all_vessel"] = mean_
    result["max_all_vessel"] = max_
    result["min_all_vessel"] = min_
    return mean_, max_
    
#def skeleton_dist(skeleton):

