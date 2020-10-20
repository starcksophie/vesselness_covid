import numpy as np

def get_analytics(img, verbose=False):
    numvox = np.count_nonzero(img)
    if verbose:
        print('Number of voxel containing vessles: ', numvox)