import sklearn.neighbors as knn

def knn_graph(n_neighbors, pcd):
    """
    Creates a k-nearest-neighbors graph from a point cloud.
    :param n_neighbors: Number of neighbors to consider.
    :param pcd: Point cloud.
    :return: A k-nearest-neighbors graph.
    """
    # Create the k-nearest-neighbors graph
    neigh = knn.KNeighborsClassifier(n_neighbors)
    neigh.fit(X, y)

