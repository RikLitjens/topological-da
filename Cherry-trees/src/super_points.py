import open3d as o3d
import numpy as np
import time
from helpers import *
from preprocess import *

# def load_pcd(path):
#     """Loads a point cloud from a file and returns it as an open3d point cloud."""
#     pcd_file = path
#     pcd = o3d.io.read_point_cloud(pcd_file)
#     return pcd

def get_super_points(data, radius):
    """Gets the super points from a point cloud.
    args:
        data (numpy array): the data from the point cloud
        radius (float): the radius to use for clustering

    returns:
        clusters (python list): the clusters of points
        super_points (numpy array): the super points
    """
    start = time.time()
    sort_idx = data[:, 0].argsort()
    sorted_data = data[sort_idx]
    mask = np.arange(len(sorted_data))
    counter = len(mask)
    super_points = []
    clusters = np.full(data.shape[0], -1, dtype=int)
    
    print("Calculating super points...")
    cluster_id = 0
    while counter > 0:
        print(f"Cluster: {cluster_id} | Points left: {counter}")

        # Get the smapled point
        sample_index = np.random.randint(len(mask)) # relative point in the mask
        masked_data = sorted_data[mask]
        current_point = masked_data[sample_index]
        
        # Build the cluster
        cluster_indices = [sample_index]
        for c, candidate_point in enumerate(masked_data):
            if c == sample_index: continue

            distance = dist(current_point, candidate_point)
            if distance <= radius:
                cluster_indices.append(c)
        
        # for c, candidate_point in enumerate(masked_data[sample_index+1:]):
        #     if c != sample_index:
        #         distance = dist(current_point, candidate_point)
        #         if distance <= radius:
        #             cluster_indices.append(sample_index+c+1)
        #         if abs(current_point[0]-candidate_point[0]) > radius:
        #             break

        # for c, candidate_point in enumerate(masked_data[sample_index-1:-1]):
        #     if c != sample_index:
        #         distance = dist(current_point, candidate_point)
        #         if distance <= radius:
        #             cluster_indices.append(sample_index-c-1)
        #         if abs(current_point[0]-candidate_point[0]) > radius:
        #             break
        
        counter -= len(cluster_indices)

        
        clusters[mask[cluster_indices]] = cluster_id
        cluster_id += 1
        super_points.append(np.mean(sorted_data[mask[cluster_indices]], axis=0))
        mask = np.delete(mask, cluster_indices)

    # Unsort the cluster order
    clusters = clusters[sort_idx.argsort()]

    end = time.time()
    print(f"Total time to calculate super points: {end-start}")

    return clusters, np.asarray(super_points)

if __name__ == "__main__":
    pcd = load_pcd("Cherry-trees/data/cloud_final_0.pcd")
    
    cleaned_pcd = clean_up(pcd)

    data = get_data(cleaned_pcd)
    _, super_array = get_super_points(data, 0.1)
    super_pcd = numpy_to_pcd(super_array)
    
    visualize(super_pcd, None)
