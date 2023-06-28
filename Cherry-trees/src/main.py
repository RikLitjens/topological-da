from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *
from persistent_homology import *
import os
from PIL import Image
from popsearch.edge import Edge
from popsearch.skeleton import LabelEnum
from popsearch.mst import *
from ripser import ripser
import gc
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# test_mst()

def prepare_edge_model():
    make_model()

# prepare model: not necessary each time, as it is saved
####prepare_edge_model()

# Get the path to the data
# local_path = get_data_path()
local_path="Cherry-trees/data/"

bag_id = 0
# Load the point cloud
pcd = load_point_cloud(local_path, bag_id, "cloud_final")
# Load the superpoints
r_super = .1
clusters, super_points = get_super_points(get_data(pcd), r_super)

# create the edges
edges = get_edges(super_points, r_super=r_super)


print("Calculate convergence to singular compleces")
pts = np.array(pcd.points)
deaths = []
print(len(edges), "edges to process")

for i, e in enumerate(edges):
    print(f"edge: {i+1}/{len(edges)}")
    t_start = time.time()
    # print("process edge:", e)
    c1 = clusters[e[0]]
    c2 = clusters[e[1]]
    c1 = get_cluster(c1, clusters, pts)
    c2 = get_cluster(c2, clusters, pts)
    # print(len(c1), len(c2))
    cu = union_of_points(c1, c2)
    # Ripser verwacht een input als (N, M) waar N>M en N,M > 0
    mx = calc_ttsc(cu)
    # print("ttsc:", mx)
    deaths.append(mx)
    # print("processed in:", time.time()-t_start, "seconds")

print(np.mean(deaths))
edge_confidences = normalize_times(deaths)


plt.show()
edges = np.array(edges)
e = super_points[edges]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(super_points[:,0], super_points[0:,1], super_points[:,2], c='green')
for i, l in enumerate(e):
    p1,p2 = l
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=(edge_confidences[i], edge_confidences[i], edge_confidences[i]))

plt.show()

# ax[0].scatter3D


# Plot the edges before and after cleaning



# TODO: remove exit and 
exit()

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
g.plot()
mst = g.kruskal()
mst.plot()
mst_cut = cut_tree(super_points, rich_edges, 0.6)
mst_cut.plot()


