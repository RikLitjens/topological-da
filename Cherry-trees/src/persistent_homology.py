import numpy as np
from numpy import random
from ripser import ripser

def union_of_points(cluster1, cluster2):
    """Creates union of two numpy arrays, can be concatenation because there are no duplicate points.

    args:
        cluster1: np array
        cluster2: np array

    return:
        np array : concatenation of two input arrays
    
    """
    union = np.vstack((cluster1, cluster2))
    x = int(union.shape[0] * 0.1)
    sampled = union[random.choice(union.shape[0],x,replace=False),:]
    return sampled

def calc_ttsc(pc) -> float:
    """
    Calculate the time to singular component in the vitoris-rips complex
    
    Args:
        point_cloud: 1D np array of all points used in the simplicial complex
    """

    H0 = ripser(pc)['dgms'][0]
    mx = np.amax(H0[:-1], axis=0)[1]
    return mx


def normalize_times(times):
    """Normalize the (persistence) times
    
    args:
        times: array-like of the different persistence times

    return:
        np array; normalized persistence times
    """
    max_time = max(times)
    min_time = min(times)
    max_diff = max_time - min_time
    normalized_times = np.array([(time-min_time)/max_diff for time in times])
    return normalized_times