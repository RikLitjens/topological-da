import open3d as o3d
import numpy as np
import time
from helpers import *
import sys
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
# Get the path to the data
    local_path = get_data_path()

    print(local_path)

    bag_id = 4
    # Load the point cloud
    pcd = load_point_cloud(local_path, bag_id, "cloud_final")

    # Load the superpoints
    clusters, super_points = get_super_points(get_data(pcd), 0.1)

    # Make point cloud out of superpoints
    super_points_pcd = numpy_to_pcd(super_points)

    # # Visualize the point cloud
    o3d.visualization.draw_geometries([super_points_pcd],
                                        zoom=0.455,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])