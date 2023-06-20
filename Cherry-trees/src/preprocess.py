import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

def load_pcd(path):
    """Loads a point cloud from a file and returns it as an open3d point cloud."""
    pcd_file = path
    pcd = o3d.io.read_point_cloud(pcd_file)
    return pcd

def get_data(pcd):
    """Gets the data from a point cloud and returns it as a numpy array."""
    data = np.asarray(pcd.points)
    return data

def numpy_to_pcd(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    return pcd

def db_cluster(pcd):
    """Clusters a point cloud using DBSCAN and returns the colors of the clusters."""
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))
    
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    return colors, labels

def get_lowest_cluster(data, max_label, labels):
    min_y = np.inf
    lowest_cluster_label = -1
    for label in range(max_label):
        masked_labels = data[labels==label]
        centroid = np.mean(masked_labels, axis=0)[2]

        if centroid < min_y:
            min_y = centroid
            lowest_cluster_label = label

    if lowest_cluster_label == -1:
        raise Exception("Could not compute lowest cluster.")

    return lowest_cluster_label
            


def visualize(pcd, colors=None):
    """Visualizes a point cloud with colors."""
    if np.all(colors != None):
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])


def clean_up(pcd):
    data = get_data(pcd)
    brc = Birch(threshold=0.1, n_clusters=5)
    brc.fit(data)

    labels = brc.predict(data)

    max_label = labels.max()

    lowest = get_lowest_cluster(data, max_label, labels)

    cleaned_data = data[np.where(labels!=lowest)]

    cleaned_pcd = numpy_to_pcd(cleaned_data)
    return cleaned_pcd


if __name__ == "__main__":
    pcd = load_pcd("Cherry-trees/data/cloud_final_0.pcd")
    data = get_data(pcd)
    brc = Birch(threshold=0.1, n_clusters=5)
    brc.fit(data)

    labels = brc.predict(data)

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0

    visualize(pcd, None)

    lowest = get_lowest_cluster(data, max_label, labels)

    cleaned_data = data[np.where(labels!=lowest)]

    cleaned_pcd = numpy_to_pcd(cleaned_data)
    
    visualize(cleaned_pcd, None)
