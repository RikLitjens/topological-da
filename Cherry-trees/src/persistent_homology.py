import numpy as np
from ripser import ripser
from helpers import *
from popsearch.skeleton_components import Edge, LabelEnum

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

def build_edge_list(edge_confidences, edges, super_points):
    """Build the edge list from the edge confidences and the edges"""
    edge_objects = []
    for i, primitive_edge in enumerate(edges):
        p_start = super_points[primitive_edge[0]]
        p_end = super_points[primitive_edge[1]]
        conf = edge_confidences[i]
        for label in LabelEnum:
            edge_objects.append(Edge(p_start, p_end, conf, label))
    return edge_objects

def calc_edge_confidences(pcd, clusters, edges):
    print("Calculate convergence to singular compleces")
    pts = np.array(pcd.points)
    deaths = []
    print(len(edges), "edges to process")

    for i, e in enumerate(edges):
        print(f"edge: {i+1}/{len(edges)}")

        # Calculate the cluster of the two points
        c1 = clusters[e[0]]
        c2 = clusters[e[1]]
        c1 = get_cluster(c1, clusters, pts)
        c2 = get_cluster(c2, clusters, pts)

        cu = union_of_points(c1, c2)

        # Calculate the time to singular component
        mx = calc_ttsc(cu)

        deaths.append(mx)

    edge_confidences = 1 - normalize_times(deaths)

    return edge_confidences

    