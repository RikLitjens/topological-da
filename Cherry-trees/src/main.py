from reebgraph.graph_computation import compute_reeb, plot_reeb
from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *
from persistent_homology import *
import os
from PIL import Image
from popsearch.skeleton_components import Edge
from popsearch.popsearch import PopSearch
from popsearch.pop_helpers import find_base_node
from popsearch.skeleton import LabelEnum
from popsearch.mst import *
from preprocess import clean_up, rotate_z_up
import pickle
<<<<<<< HEAD
from strategies import *

=======

# test_mst()
>>>>>>> origin/reeb-graph

# Strategy to use
strat = "CNN"
# strat = "homology"
# strat = "reeb"

<<<<<<< HEAD
#############################
# Point cloud preprocessing #
#############################

local_path = get_data_path()
bag_id = 0

# Load the point cloud
pcd = load_point_cloud(local_path, bag_id, "cloud_final")
pcd = rotate_z_up(pcd)
pcd = clean_up(pcd)

################
# CNN strategy #
################

if strat == "CNN":
    strat_CNN(pcd, prepped_model=True, bag_id=bag_id)

#############################
# Persistent homology strat #
#############################

if strat == "homology":
    strat_persistent_homology(pcd)
=======
# prepare model: not necessary each time, as it is saved
####prepare_edge_model()

# Get the path to the data
local_path = get_data_path()

bag_id = 2
# Load the point cloud
pcd = rotate_z_up(load_point_cloud(local_path, bag_id, "cloud_final"))
# clusters, super_points = get_super_points(get_data(pcd), 0.05)
# print(len(super_points))
# visualize_point_cloud(super_points)
>>>>>>> origin/reeb-graph

#######################
# Reeb graph strategy #
#######################

<<<<<<< HEAD
if strat == "reeb":
    strat_reeb_graph(pcd)
=======
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
>>>>>>> origin/reeb-graph

# bag_id = 0
# # Load the point cloud
# pcd = load_point_cloud(local_path, bag_id, "cloud_final")
# pcd = rotate_z_up(pcd)
# pcd = clean_up(pcd)

# # get tips
# g = Graph(super_points, rich_edges)
# g.plot()-
# mst = g.kruskal()
# mst.plot()
# mst_cut = cut_tree(super_points, rich_edges, 0.6)
# mst_cut.plot()


<<<<<<< HEAD
# Load the inputs from the pickle file
# with open("popsearch_inputs.pickle", "rb") as file:
#     super_points, edge_objects, tree_tips, base = pickle.load(file)

# for ed in edge_objects:
#     ed.conf = ed.conf.item()


# base_point = find_base_node(super_points)


# # Do the pop search
# ps = PopSearch(super_points, edge_objects, tree_tips, base_point)
# ps.do_pop_search()
=======
pcd = clean_up(pcd)
reeb = compute_reeb(get_data(pcd), 0.1, 0.015)
plot_reeb(reeb)
>>>>>>> origin/reeb-graph
