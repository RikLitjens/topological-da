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
from ripser import ripser
import gc
from numpy import random
import matplotlib.pyplot as plt
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
clusters, super_points = get_super_points(get_data(pcd), 0.1)
del pcd
gc.collect()

# create the edges
edges = get_edges(super_points, 0.1)
# edge_histograms = edge_evaluation(edges, super_points, clusters, 0.1, bag_id)

# Determine the edge confidences with persistent homology

# For each edge
# 1. Get union of points in clusters
# 2. Get time it takes to become a single component

# normalize the radia

def union_of_points(cluster1, cluster2):
    """Creates union of two numpy arrays, can be concatenation because there are no duplicate points.

    args:
        cluster1: np array
        cluster2: np array

    return:
        np array : concatenation of two input arrays
    
    """
    union = np.vstack((cluster1, cluster2))
    x = int(union.shape[0] * 0.2)
    sampled = union[random.choice(union.shape[0],x,replace=False),:]
    return sampled

def calc_ttsc(pc) -> float:
    """
    Calculate the time to singular component in the vitoris-rips complex
    
    Args:
        point_cloud: 1D np array of all points used in the simplicial complex
    """

    H0 = ripser(pc)['dgms'][0]
    mx = np.amax(H0[:-1], axis=0)[1]
    return mx


def normalize_times(times):
    """Normalize the (persistence) times
    
    args:
        times: array-like of the different persistence times

    return:
        np array; normalized persistence times
    """
    max_time = max(times)
    min_time = min(times)
    max_diff = max_time = min_time
    normalized_times = np.array([(time-min_time)/max_diff for time in times])
    return normalized_times

deaths = []
print("Calculate convergence to singular compleces")
print(len(edges), "edges to process")

for i, e in enumerate(edges):
    print(f"edge: {i+1}/{len(edges)}")
    t_start = time.time()
    # print("process edge:", e)
    c1 = clusters[e[0]]
    c2 = clusters[e[1]]
    # print(len(c1), len(c2))
    cu = union_of_points(c1, c2)
    # Ripser verwacht een input als (N, M) waar N>M en N,M > 0
    mx = calc_ttsc(cu)
    # print("ttsc:", mx)
    deaths.append(mx)
    # print("processed in:", time.time()-t_start, "seconds")

print(np.mean(deaths))
normalized_times = normalize_times(deaths)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(deaths, bins=20)
axs[1].hist(normalized_times, bins=20)

plt.show()


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


