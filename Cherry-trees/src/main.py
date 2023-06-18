import open3d as o3d
import helpers
import configparser
from super_points import *

config = configparser.RawConfigParser()
config.read(r"Cherry-trees\config\config.ini")
local_path = config.get('DATA', 'PATH')

print(local_path)

# Load the point cloud
pcd = load_pcd(local_path+"cloud_final.pcd")

# Load the superpoints
clusters, super_points = get_super_points(get_data(pcd), 0.1)

# Make point cloud out of superpoints
super_points_pcd = numpy_to_pcd(super_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([super_points_pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])