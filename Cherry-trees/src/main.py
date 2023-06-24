from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *
import os
from PIL import Image
from popsearch.edge import Edge
from popsearch.skeleton import LabelEnum
from popsearch.mst import *

# test_mst()

def prepare_edge_model():
    make_model()

# prepare model: not necessary each time, as it is saved
####prepare_edge_model()

# Get the path to the data
local_path = get_data_path()

bag_id = 0
# Load the point cloud
pcd = load_point_cloud(local_path, bag_id, "cloud_final")

# o3d.visualization.draw_geometries([pcd],
#                                     zoom=0.455,
#                                     front=[-0.4999, -0.1659, -0.8499],
#                                     lookat=[2.1813, 2.0619, 2.0999],
#                                     up=[0.1204, -0.9852, 0.1215])

# Load the superpoints
clusters, super_points = get_super_points(get_data(pcd), 0.1)

# # Make point cloud out of superpoints
# super_points_pcd = numpy_to_pcd(super_points)

# # # Visualize the point cloud
# o3d.visualization.draw_geometries([super_points_pcd],
#                                     zoom=0.455,
#                                     front=[-0.4999, -0.1659, -0.8499],
#                                     lookat=[2.1813, 2.0619, 2.0999],
#                                     up=[0.1204, -0.9852, 0.1215])

# create the edges
edges = get_edges(super_points, 0.1)
edge_histograms = edge_evaluation(edges, super_points, clusters, 0.1, bag_id)


# Load the saved model
model = NET()
model_path = os.path.join(os.getcwd(), 'Cherry-trees/src/deepnet/model.tree')
model.load_state_dict(torch.load(model_path))

X = torch.tensor(edge_histograms).reshape(-1, 1, 32, 16).float()

edge_confidences = model(X)

# save some examples
# for i in range(5):
#     img = Image.fromarray(edge_histograms[i].astype(np.uint8), 'L')
#     img.save(f"example-{edge_confidences[i]}.png".replace(".", "_", 1))


# convert to edge class
print(len(super_points))
print(len(edges))

rich_edges = []
for i, primitive_edge in enumerate(edges):
    p_start = super_points[primitive_edge[0]]
    p_end = super_points[primitive_edge[1]]
    conf = edge_confidences[i]
    for label in LabelEnum:
        rich_edges.append(Edge(p_start, p_end, conf, label))


# get tips
g = Graph(super_points, rich_edges)
print(f'There are {len(g.find_connected_components(True))} components')

g.plot()
mst = g.kruskal()
# mst.plot()
mst_cut_tree, _ = cut_tree(super_points, rich_edges, 0.6)
tree_tips = mst_cut_tree.find_tree_tips()
print('Tree tips')
mst_cut_tree.plot(tips=tree_tips)


