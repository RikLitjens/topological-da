from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *
import os
from PIL import Image
from popsearch.skeleton_components import Edge
from popsearch.popsearch import PopSearch
from popsearch.pop_helpers import find_base_node
from popsearch.skeleton import LabelEnum
from popsearch.mst import *
from preprocess import clean_up, rotate_z_up
import pickle

# # test_mst()


# def prepare_edge_model():
#     make_model()


# # prepare model: not necessary each time, as it is saved
# ####prepare_edge_model()

# # Get the path to the data
# local_path = get_data_path()

# bag_id = 0
# # Load the point cloud
# pcd = load_point_cloud(local_path, bag_id, "cloud_final")
# pcd = rotate_z_up(pcd)
# pcd = clean_up(pcd)

# # Create the coordinate frame mesh
# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# # Combine the point cloud and coordinate frame into a single geometry list
# geometries = [pcd, coord_frame]

# # # Visualize the geometries with axis lines
# # o3d.visualization.draw_geometries(
# #     geometries,
# #     zoom=0.455,
# #     front=[-0.4999, -0.1659, -0.8499],
# #     lookat=[2.1813, 2.0619, 2.0999],
# #     up=[0.1204, -0.9852, 0.1215],
# # )

# # Load the superpoints
# clusters, super_points = get_super_points(get_data(pcd), 0.1)

# # # Make point cloud out of superpoints
# # super_points_pcd = numpy_to_pcd(super_points)

# # # # Visualize the point cloud
# # geometries = [super_points_pcd, coord_frame]
# # o3d.visualization.draw_geometries(geometries,
# #                                     zoom=0.455,
# #                                     front=[-0.4999, -0.1659, -0.8499],
# #                                     lookat=[2.1813, 2.0619, 2.0999],
# #                                     up=[0.1204, -0.9852, 0.1215])

# # create the edges
# edges = get_edges(super_points, 0.1)
# edge_histograms = edge_evaluation(edges, super_points, clusters, 0.1, bag_id)


# # Load the saved model
# model = NET()
# model_path = os.path.join(os.getcwd(), "Cherry-trees/src/deepnet/model.tree")
# model.load_state_dict(torch.load(model_path))

# X = torch.tensor(edge_histograms).reshape(-1, 1, 32, 16).float()

# edge_confidences = model(X)

# # save some examples
# # for i in range(5):
# #     img = Image.fromarray(edge_histograms[i].astype(np.uint8), 'L')
# #     img.save(f"example-{edge_confidences[i]}.png".replace(".", "_", 1))


# # convert to edge class
# print(len(super_points))
# print(len(edges))

# edge_objects = []
# for i, primitive_edge in enumerate(edges):
#     p_start = super_points[primitive_edge[0]]
#     p_end = super_points[primitive_edge[1]]
#     conf = edge_confidences[i]
#     for label in LabelEnum:
#         edge_objects.append(Edge(p_start, p_end, conf, label))


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
with open("popsearch_inputs.pickle", "rb") as file:
    super_points, edge_objects, tree_tips, base = pickle.load(file)

for ed in edge_objects:
    ed.conf = ed.conf.item()


base_point = find_base_node(super_points)


# Do the pop search
ps = PopSearch(super_points, edge_objects, tree_tips, base_point)
ps.do_pop_search()
