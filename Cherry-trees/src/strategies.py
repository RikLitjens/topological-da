from persistent_homology import *
from popsearch.popsearch import PopSearch
from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *
from popsearch.mst import *
from operator import itemgetter
from reebgraph.graph_computation import compute_reeb, plot_reeb
import time

def strat_CNN(pcd, prepped_model=False, bag_id=0):
    """
    Strategy for a calculating the skeleton with a CNN
    This is the baseline strategy

    Args:
        pcd: open3d point cloud
        prepped_model: boolean, whether or not there already exists a model
        bag_id: int, the bag id of the point cloud
    """
    start = time.time()

    # Calculate the superpoints and corresponding clusters
    clusters, super_points = get_super_points(get_data(pcd), 0.1)

    # Calculate the MST
    edges = get_edges(super_points, 0.1)

    # Calculate the edge histograms
    edge_histograms = edge_evaluation(edges, super_points, clusters, 0.1, bag_id, pcd)

    # Create the CNN model if none is provided
    if not prepped_model:
        make_model()

    # Load the CNN model
    model = NET()
    model_path = os.path.join(os.getcwd(), "Cherry-trees/src/deepnet/model.tree")
    model.load_state_dict(torch.load(model_path))

    # Calculate the edge confidences
    X = torch.tensor(edge_histograms).reshape(-1, 1, 32, 16).float()
    edge_confidences = model(X).detach().numpy()

    time2ec = time.time() - start

    # convert to edge class
    edge_list = build_edge_list(edge_confidences, edges, super_points)
    # visualize_edge_confidences(edge_list, super_points, name="cnn_edge_conf")

    # Create the graph
    G = Graph(super_points, edge_list)

    # Calculate the connected components
    cc = G.find_connected_components(True)

    # Determine lowest point of largest connected component
    lengths = [len(c) for c in cc]
    lowest = min(cc[np.argmax(lengths)], key=itemgetter(2))

    mst_cut_tree, _ = cut_tree(super_points, edge_list, 0.3)
    tree_tips = mst_cut_tree.find_tree_tips()

    # Do the pop search
    ps = PopSearch(super_points, edge_list, tree_tips, base_node=lowest)
    ps.do_pop_search()
    total_time = time.time() - start
    print(f"time to edge confidence = {time2ec} total time = {total_time}")


def strat_persistent_homology(pcd):
    """
    Strategy for a calculating the skeleton with persistent homology

    Args:
        pcd: open3d point cloud
    """
    start = time.time()
    # Calculate the superpoints and corresponding clusters
    clusters, super_points = get_super_points(get_data(pcd), 0.1)

    # Calculate the MST
    edges = get_edges(super_points, 0.1)

    # Calculate the edge confidences
    edge_conf = calc_edge_confidences(pcd, clusters, edges)

    time2ec = time.time() - start
    # Convert to edge class
    edge_list = build_edge_list(edge_conf, edges, super_points)
    # visualize_edge_confidences(edge_list, super_points, name="homology_edge_conf")

    # Create the graph
    G = Graph(super_points, edge_list)

    # Calculate the connected components
    cc = G.find_connected_components(True)

    # Determine lowest point of largest connected component
    lengths = [len(c) for c in cc]
    lowest = min(cc[np.argmax(lengths)], key=itemgetter(2))

    mst_cut_tree, _ = cut_tree(super_points, edge_list, 0.6)
    tree_tips = mst_cut_tree.find_tree_tips()

    # Do the pop search
    ps = PopSearch(super_points, edge_list, tree_tips, base_node=lowest)
    ps.do_pop_search()

    total_time = time.time() - start
    print(f"time to edge confidence = {time2ec} total time = {total_time}")


def strat_reeb_graph(pcd):
    """
    Strategy for a calculating the skeleton with a reeb graph

    Args:
        pcd: open3d point cloud
    """
    start = time.time()
    reeb = compute_reeb(get_data(pcd), 0.1, 0.015)
    plot_reeb(reeb)
    print(f"total time = {time.time()-start}")

