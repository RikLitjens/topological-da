import open3d as o3d
import numpy as np
import time
from helpers import *

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
    sorted_data = data[data[:, 0].argsort()]
    mask = np.full(len(sorted_data), True, dtype=bool)
    indices = np.arange(0, len(sorted_data))
    counter = len(mask)

    super_points = []
    clusters = []
    
    print("Calculating super points...")
    
    while counter > 0:
        print(f" Points left: {counter}")
        sample_index = np.random.choice(indices[mask], 1)
        current_point = sorted_data[sample_index][0]

        cluster = [current_point]
        cluster_indices = [sample_index[0]]

        for c, candidate_point in enumerate(sorted_data[mask][sample_index[0]+1:]):
            if c != sample_index:
                if len(current_point) != 3:
                    print(f"FAILURE! Current point: {current_point}")
                if len(candidate_point) != 3:
                    print(f"FAILURE! Candidate point: {candidate_point}")
                distance = dist(current_point, candidate_point)
                if distance <= radius:
                    cluster.append(candidate_point)
                    cluster_indices.append(sample_index[0]+c+1)
                if abs(current_point[0]-candidate_point[0]) > radius:
                    break

        for c, candidate_point in enumerate(sorted_data[mask][sample_index[0]-1:-1]):
            if c != sample_index:
                distance = dist(current_point, candidate_point)
                if distance <= radius:
                    cluster.append(candidate_point)
                    cluster_indices.append(sample_index[0]-c-1)
                if abs(current_point[0]-candidate_point[0]) > radius:
                    break
        
        counter -= len(cluster)

        mask[mask][cluster_indices] = False
        clusters.append(cluster)
        super_points.append(np.mean(cluster, axis=0))

    end = time.time()
    print(f"Total time to calculate super points: {end-start}")

    return clusters, np.asarray(super_points)

def load_super_points(path):
    """Loads the super points from a file and returns them as a numpy array."""
    pcd = load_pcd(path)
    data = get_data(pcd)
    return get_super_points(data, 0.1)

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