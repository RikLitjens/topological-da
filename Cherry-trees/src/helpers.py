import open3d as o3d

def load_point_cloud(local_path, bag_id, pointcloud_name):
    pcd = o3d.io.read_point_cloud(f"{local_path}/bag_{bag_id}/{pointcloud_name}.pcd")
    return pcd

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd], 
                                      zoom=0.3412, 
                                      front=[0.4257, -0.2125, -0.8795], 
                                      lookat=[2.6172, 2.0475, 1.532], 
                                      up=[-0.0694, -0.9768, 0.2024])