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
from strategies import *


# Strategy to use
strat = "CNN"
# strat = "homology"
# strat = "reeb"

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

#######################
# Reeb graph strategy #
#######################

if strat == "reeb":
    strat_reeb_graph(pcd)


# # get tips
# g = Graph(super_points, edge_objects)
# print(f"There are {len(g.find_connected_components(True))} components")

# # g.plot()
# mst = g.kruskal()
# # mst.plot()
# mst_cut_tree, _ = cut_tree(super_points, edge_objects, 0.6)
# tree_tips = mst_cut_tree.find_tree_tips()
# # mst_cut_tree.plot(tips=tree_tips)

# base = tree_tips[0]
# # Load inputs to make quicker
# with open("popsearch_inputs.pickle", "wb") as file:
#     pickle.dump((super_points, edge_objects, tree_tips, tree_tips[0]), file)

# Load the inputs from the pickle file
# with open("popsearch_inputs.pickle", "rb") as file:
#     super_points, edge_objects, tree_tips, base = pickle.load(file)

# for ed in edge_objects:
#     ed.conf = ed.conf.item()


# base_point = find_base_node(super_points)


# # Do the pop search
# ps = PopSearch(super_points, edge_objects, tree_tips, base_point)
# ps.do_pop_search()
