import numpy as np
from copy import deepcopy
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize

global result

def distance(seg, verbose=False):
    """
    Computes the distance map of a segmentation.
    
    seg: numpy.ndarray : segmented array
    verbose: bool : display log option.

    return: numpy.ndarray : a distance map.
    """
    #  compute distance map
    if verbose:
        print("computing distance map...")
    distance_map = ndi.distance_transform_edt(seg)
    if verbose:
        plt.imshow(distance_map[185])
    return distance_map

def label_value(dist):
    """
        Aggregates distances for each label.

        dist: numpy.ndarray : distance map of the vessel segmentation.

        return: numpy.ndarray : all the distances for each label.
    """
    label_map, label_nbr = label(dist, return_num=True)

    result["component_count"] = label_nbr

    print(label_nbr, flush=True)
    label_map = label_map.flatten()
    dist_size = int(len(label_map) / 10000)
    print("dist_size", dist_size)

    dist = dist.flatten()
    dist_per_label = [[] for i in range(label_nbr+1)];
    for i in range(0, len(label_map)-dist_size, dist_size):
        label_map_i = label_map[i:i + dist_size]
        flat_dist_i = dist[i : i + dist_size]
        for k in np.unique(label_map_i): 
            k = int(k)
            if k == 0: # not a valid label
                continue
            #sub =  np.where(label_map_i == k, label_map_i * 0, flat_dist_i )
            sub = flat_dist_i[(label_map_i == k)]
            dist_per_label[k] += list(sub) # append labels to list 

    dist_per_label = np.array([np.array(x, dtype=np.float64) for x in dist_per_label], dtype=object)
    print("after the for in label_value\n", flush=True)
    return dist_per_label


def get_analytics(img, mask, dist_per_label, verbose=False):
    """
        Computes the metrics on all the labels, and fills the result array.

        img: numpy.ndarray : segmented array.
        mask: numpy.ndarray : mask array of the associated img.
        dist_per_label: numpy.ndarray : distance array per label.
        verbose: bool : display log option.
    """
    numvox = np.count_nonzero(img)
    result["voxel_nbr"] = numvox
    result["lung_pixels"] = np.count_nonzero(mask)
    if verbose:
        print('Number of voxel containing vessles: ', numvox)

    f_mean = np.vectorize(lambda x : x.mean() if x.any() else None)
    mean_ = f_mean(dist_per_label)
    mean_ = mean_[~np.isnan(mean_.astype(np.float64))] #remove the Nones
    f_max = np.vectorize(lambda x : x.max() if x.any() else None)
    max_ = f_max(dist_per_label)
    max_ = max_[~np.isnan(max_.astype(np.float64))] #remove the Nones
    result["mean_mean_all_vessel"] = mean_.mean() if mean_.any() else None
    result["std_deviation"] = np.std(mean_) if mean_.any() else None
    result["max_max_all_vessel"] = max_.max() if max_.any() else None
    result["min_max_all_vessel"] = max_.min() if max_.any() else None
    result["mean_max_all_vessel"] = max_.mean() if max_.any() else None
    result["mean_all_vessel"] = mean_
    print("most computations are done\n", flush=True)
    result["max_all_vessel"] = max_
