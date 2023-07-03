import open3d as o3d
import math
import numpy as np
import configparser
from PIL import Image
from random import random

def get_data_path():
    config = configparser.RawConfigParser()
    config.read(r"Cherry-trees/config/config.ini")
    return config.get("DATA", "PATH")


def load_point_cloud(local_path, bag_id, pointcloud_name):
    pcd = o3d.io.read_point_cloud(rf"{local_path}/bag_{bag_id}/{pointcloud_name}.pcd")
    return pcd


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[0, 0, 0],
        up=[0.4694, -0.9768, 0.2024],
    )


def get_data(pcd):
    """Gets the data from a point cloud and returns it as a numpy array."""
    data = np.asarray(pcd.points)
    return data


def numpy_to_pcd(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    return pcd


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def get_image(path):
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
    sampled = union[random.choice(union.shape[0],x,replace=False),:]
    return sampled

def get_cluster(cluster_id, clusters, points):
    return points[np.where(clusters == cluster_id)]