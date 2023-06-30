from reebgraph.graph_computation import compute_reeb
from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *
import os
from PIL import Image
from popsearch.skeleton_components import Edge
from popsearch.popsearch import PopSearch
from popsearch.skeleton import LabelEnum
from popsearch.mst import *
from preprocess import clean_up, rotate_z_up
import pickle

# test_mst()

# def prepare_edge_model():
#     make_model()

# prepare model: not necessary each time, as it is saved
####prepare_edge_model()

# Get the path to the data
local_path = get_data_path()

bag_id = 2
# Load the point cloud
pcd = load_point_cloud(local_path, bag_id, "cloud_final")
# clusters, super_points = get_super_points(get_data(pcd), 0.05)
# print(len(super_points))
# visualize_point_cloud(super_points)

# # Load the superpoints
# clusters, super_points = get_super_points(get_data(pcd), 0.1)

# # create the edges
# edges = get_edges(super_points, 0.1)
# edge_histograms = edge_evaluation(edges, super_points, clusters, 0.1, bag_id)


# # Load the saved model
# model = NET()
# model_path = os.path.join(os.getcwd(), 'Cherry-trees/src/deepnet/model.tree')
# model.load_state_dict(torch.load(model_path))

# X = torch.tensor(edge_histograms).reshape(-1, 1, 32, 16).float()

# edge_confidences = model(X)

# save some examples
# for i in range(5):
#     img = Image.fromarray(edge_histograms[i].astype(np.uint8), 'L')
#     img.save(f"example-{edge_confidences[i]}.png".replace(".", "_", 1))


# # convert to edge class
# print(len(super_points))
# print(len(edges))

# rich_edges = []
# for i, primitive_edge in enumerate(edges):
#     p_start = super_points[primitive_edge[0]]
#     p_end = super_points[primitive_edge[1]]
#     conf = edge_confidences[i]
#     for label in LabelEnum:
#         rich_edges.append(Edge(p_start, p_end, conf, label))

# bag_id = 0
# # Load the point cloud
# pcd = load_point_cloud(local_path, bag_id, "cloud_final")
# pcd = rotate_z_up(pcd)
# pcd = clean_up(pcd)

# # get tips
# g = Graph(super_points, rich_edges)
# g.plot()
# mst = g.kruskal()
# mst.plot()
# mst_cut = cut_tree(super_points, rich_edges, 0.6)
# mst_cut.plot()



compute_reeb(get_data(pcd), 0.3, 0.05)
