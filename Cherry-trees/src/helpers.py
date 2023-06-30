from matplotlib import pyplot as plt
import open3d as o3d
import math
import numpy as np
import configparser
from PIL import Image


def get_data_path():
    config = configparser.RawConfigParser()
    config.read(r"Cherry-trees\config\config.ini")
    return config.get("DATA", "PATH")


def load_point_cloud(local_path, bag_id, pointcloud_name):
    pcd = o3d.io.read_point_cloud(rf"{local_path}\bag_{bag_id}\{pointcloud_name}.pcd")
    return pcd


def visualize_point_cloud(pcd):
    # o3d.visualization.draw_geometries(pcd, 
    #                                   zoom=0.3412, 
    #                                   front=[0.4257, -0.2125, -0.8795], 
    #                                   lookat=[0, 0, 0], 
    #                                   up=[0.4694, -0.9768, 0.2024]
    
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in pcd:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.3, 0.3, 0.3])
    viewer.run()
    viewer.destroy_window()

def visualize_point_cloud_scatter(pcd):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='r', marker='o')
    plt.show()

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

def choose_f():
    def f(x, y, z):
        return z
    return f