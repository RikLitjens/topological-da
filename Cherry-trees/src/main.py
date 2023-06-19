from super_points import *
from helpers import *
from preprocessing import *


# Get the path to the data
local_path = get_data_path()

print(local_path)

# Load the point cloud
pcd = load_point_cloud(local_path, 0, "cloud_final")

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
edges = []
# Define the edges
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        if dist(points[i], points[j]) <= 2 * r_super:
            edges.append(i, j)
            edges.append(j, i)

n_i = points[edges[0][0]]
n_j = points[edges[0][1]]

cluster_i = clusters[edges[0][0]]
cluster_j = clusters[edges[0][1]]
total_cluster = cluster_k.extend(cluster_j)

# X-axis is equal to the direction of the edge
x_axis_new = normalize(n_j - n_i)

square_cluster = np.matmul(np.matrix.transpose(total_cluster), total_cluster)
U, S, Vh = np.linalg.svd(square_cluster)

min_eigenvalues = np.argsort(S)[-3:]
min_eigenvalue_index = min_eigenvalues[0]

# The z-axis is equal to the 3rd least significant comoponent (eigenvector belonging to the third lowest eigenvalues)
z_axis_new = U[min_eigenvalue_index]

# The y-axis is then perpendicular to the x_axis_new - z_axis_new plane
y_axis_new = np.cross(z_axis_new, x_axis_new)


# Find rotation from x, y, z to x_axis_new, y_axis_new, a_axis_new
# Because the axes are unit vectors (1, 0, 0), (0, 1, 0) and (0, 0, 1), 
# the rotation matrix is equal to the new axes in order, transposed due to it being a rotation from new axes to old axes
rotation_matrix = np.matrix.transpose(np.array([x_axis_new, y_axis_new, z_axis_new]))

x_axis = np.matmul(rotation_matrix, x_axis_new)
y_axis = np.matmul(rotation_matrix, y_axis_new)
z_axis = np.matmul(rotation_matrix, z_axis_new)

print(x_axis)
print(y_axis)
print(z_axis)

# for i in range(len(total_cluster)):
#     total_cluster[i] = np.matmul(rotation_matrix, total_cluster[i])