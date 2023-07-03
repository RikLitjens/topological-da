from persistent_homology import *
from super_points import *
from helpers import *
from edge_preprocessing import *
from deepnet.neuralnet import *
from deepnet.net_main import *

def strat_CNN(pcd, prepped_model=None, bag_id=0):
    """ 
    Strategy for a calculating the skeleton with a CNN
    This is the baseline strategy

    Args:
        pcd: open3d point cloud
    """

    # Calculate the superpoints and corresponding clusters
    clusters, super_points = get_super_points(get_data(pcd), 0.1)

    # Calculate the MST
    edges = get_edges(super_points, 0.1)

    # Calculate the edge histograms
    edge_histograms = edge_evaluation(edges, super_points, clusters, 0.1, bag_id)


    # Create the CNN model if none is provided
    if prepped_model is None:
        make_model()
    
    # Load the CNN model
    model = NET()
    model_path = os.path.join(os.getcwd(), "Cherry-trees/src/deepnet/model.tree")
    model.load_state_dict(torch.load(model_path))

    X = torch.tensor(edge_histograms).reshape(-1, 1, 32, 16).float()
    edge_confidences = model(X)

    # convert to edge class
    edge_list = build_edge_list(edge_confidences, edges, super_points)

    # TODO: Create the tree based on constraints

    pass

def strat_persistent_homology(pcd):
    """ 
    Strategy for a calculating the skeleton with persistent homology
    
    Args:
        pcd: open3d point cloud
    """
    # Calculate the superpoints and corresponding clusters
    clusters, super_points = get_super_points(get_data(pcd), 0.1)

    # Calculate the MST
    edges = get_edges(super_points, 0.1)

    # Calculate the edge confidences
    edge_conf =  calc_edge_confidences(pcd, clusters, edges)

    # Convert to edge class
    egde_list = build_edge_list(edge_conf, edges, super_points)

    # TODO: Create the tree based on constraints

def strat_reeb_graph(pcd):
    """
    Strategy for a calculating the skeleton with a reeb graph
    
    Args:
        pcd: open3d point cloud
    """
    pass