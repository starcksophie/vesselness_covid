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
    dist_size = int(len(label_map.flatten()) / 100000)
    print(dist_size, flush=True)
    label_map = label_map.flatten()
    dist = dist.flatten()
    #dist_per_label = np.array([np.array([
    #    flat_dist[i]  for i in range(len(flat_dist)) if label_map[i] == n ])
    #                    for n in range(1, label_nbr)])
    dist_per_label = np.array([np.array([]) for i in range(label_nbr)]);
    for i in range(dist_size):
        label_map_i = label_map[i:i + dist_size]
        flat_dist_i = dist[i : i + dist_size]
        print("plop" , np.unique(label_map_i))
        for k in np.unique(label_map_i): 
            k = int(k)
            if k == 0:
                continue
            sub =  np.where(label_map_i == k, label_map_i, flat_dist_i)
            dist_per_label[k] = np.concatenate((dist_per_label[k], sub)) 
    print("after the for in label_value\n", flush=True)
    #mean_ = [a.mean() for a in dist_per_label]
    f_mean = lambda x : x.mean() if x else None;
    mean_ = f_mean(dist_per_label);
    print("mean", flush=True)
    f_max = lambda x : x.max() if x else None;
    max_ = f_max(dist_per_label);
    #max_ = [a.max() for a in dist_per_label]
    f_min = lambda x : x.min() if x else None;
    min_ = f_min(dist_per_label);
    #min_ = [a.min() for a in dist_per_label]
    print("max min", flush=True)
    result["mean_mean_all_vessel"] = mean_.mean() if mean_ else None
    result["std_deviation"] = np.std(mean_) if mean_ else None
    result["max_max_all_vessel"] = max_.max() if max_ else None
    result["mean_max_all_vessel"] = max_.mean() if max_ else None
    result["mean_all_vessel"] = mean_
    print("most computations are done\n", flush=True)
    result["max_all_vessel"] = max_
    result["min_all_vessel"] = min_
    result["component_count"] = label_nbr
    return mean_, max_
    
#def skeleton_dist(skeleton):

