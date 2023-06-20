from super_points import *
from helpers import *
from preprocessing import *
from neuralnet import *


# # Get the path to the data
# local_path = get_data_path()

# print(local_path)

# bag_id = 4
# # Load the point cloud
# pcd = load_point_cloud(local_path, bag_id, "cloud_final")

# # Load the superpoints
# clusters, super_points = get_super_points(get_data(pcd), 0.1)

# # Make point cloud out of superpoints
# super_points_pcd = numpy_to_pcd(super_points)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([super_points_pcd],
#                                     zoom=0.455,
#                                     front=[-0.4999, -0.1659, -0.8499],
#                                     lookat=[2.1813, 2.0619, 2.0999],
#                                     up=[0.1204, -0.9852, 0.1215])

# edge_evaluation(super_points, clusters, 0.10, bag_id)

path = fr"Cherry-trees\images\Training\bag0histogram_0.png"
histogram = get_image(path)
histograms = explode_data(histogram, 10)
for i in range(len(histograms)):
    img = Image.fromarray(histograms[i].astype(np.uint8), 'L')
    img.save(fr"Cherry-trees\images\exploded\bag{0}histogram{0}resample{i}.png")