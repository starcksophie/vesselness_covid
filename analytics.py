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
    #  compute distance map
    if verbose:
        print("computing distance map...")
    # distance_map = seg
    # skeleton = deepcopy(distance_map)
    # for i, slice in enumerate(distance_map):
    #     distance_map[i] = ndi.distance_transform_edt(seg[i])
        #skeleton[i] = skeletonize(seg[i])
    distance_map = ndi.distance_transform_edt(seg)
    if verbose:
        plt.imshow(distance_map[185])
    return distance_map 


def label_value(dist):
    label_map, label_nbr = label(dist, return_num=True)
    print(label_nbr, flush=True)
    label_nbr = int(label_nbr / 4)
    label_map = label_map.flatten()[label_nbr:]
    flat_dist = dist.flatten()[label_nbr:]
    dist_per_label = np.array([np.where(label_map == n, 
        label_map, flat_dist) for n in range(1, label_nbr )]) 
    print("after the for in label_value\n", flush=True)

    f_mean = lambda x : x.mean();
    mean_ = f_mean(dist_per_label);
    print("mean", flush=True)
    f_max = lambda x : x.max();
    max_ = f_max(dist_per_label);
    f_min = lambda x : x.min();
    min_ = f_min(dist_per_label);
    print("max min", flush=True)
    result["mean_mean_all_vessel"] = mean_.mean()
    result["std_deviation"] = np.std(mean_)
    result["max_max_all_vessel"] = max_.max()
    result["mean_max_all_vessel"] = max_.mean()
    result["mean_all_vessel"] = mean_
    print("most computations are done\n", flush=True)
    result["max_all_vessel"] = max_
    result["min_all_vessel"] = min_
    result["component_count"] = label_nbr
    return mean_, max_

