import open3d as o3d
import math
import numpy as np
import configparser
from PIL import Image
from numpy import random
import matplotlib.pyplot as plt


def get_data_path():
    """Returns the path to the data folder. as defined in config.ini"""
    config = configparser.RawConfigParser()
    config.read(r"Cherry-trees/config/config.ini")
    return config.get("DATA", "PATH")


def load_point_cloud(local_path, bag_id, pointcloud_name):
    """Loads a point cloud from a .pcd file."""
    pcd = o3d.io.read_point_cloud(rf"{local_path}/bag_{bag_id}/{pointcloud_name}.pcd")
    return pcd


def visualize_point_cloud(pcd):
    """Visualizes a point cloud."""
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries = [pcd, coord_frame]
    o3d.visualization.draw_geometries(
        geometries,
        zoom=0.455,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[2.1813, 2.0619, 2.0999],
        up=[0.1204, -0.9852, 0.1215],
    )


def visualize_edge_confidences(pcd, edges, edge_confidences, super_points):
    _edges = np.array(edges)
    e = super_points[_edges]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter3D(super_points[:, 0], super_points[0:, 1], super_points[:, 2], c="green")
    for i, l in enumerate(e):
        p1, p2 = l
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=(edge_confidences[i], edge_confidences[i], edge_confidences[i]),
        )

    plt.show()


def save_histogram_examples(edge_histograms, edge_confidences):
    """Saves the first 5 histograms as images."""
    for i in range(5):
        img = Image.fromarray(edge_histograms[i].astype(np.uint8), "L")
        img.save(f"example-{edge_confidences[i]}.png".replace(".", "_", 1))


def get_data(pcd):
    """Gets the data from a point cloud and returns it as a numpy array."""
    data = np.asarray(pcd.points)
    return data


def numpy_to_pcd(data):
    """Converts a numpy array to a point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    return pcd


def dist(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def get_image(path):
    """Gets an image from a path and returns it as a numpy array."""
    img = Image.open(path)
    histogram = np.array(img.getdata())
    histogram = histogram.reshape((32, 16))
    return histogram


def union_of_points(cluster1, cluster2):
    """Creates union of two numpy arrays, can be concatenation because there are no duplicate points.

    args:
        cluster1: np array
        cluster2: np array

    return:
        np array : concatenation of two input arrays

    """
    union = np.vstack((cluster1, cluster2))
    x = int(union.shape[0] * 0.25)
    sampled = union[random.choice(union.shape[0], x, replace=False), :]
    return sampled


def get_cluster(cluster_id, clusters, points):
    mask = np.where(clusters == cluster_id)
    return points[mask]
