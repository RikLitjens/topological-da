import sklearn.neighbors as knn
from rtreelib import RTree, Rect
from rtree import index as rtindex


# def knn_graph(n_neighbors, pcd):
#     """
#     Creates a k-nearest-neighbors graph from a point cloud.
#     :param n_neighbors: Number of neighbors to consider.
#     :param pcd: Point cloud.
#     :return: A k-nearest-neighbors graph.
#     """
#     # Create the k-nearest-neighbors graph
#     neigh = knn.KNeighborsClassifier(n_neighbors)
#     neigh.fit(X, y)

# class KD:
#     def __init__(self, strip):
#         self.tree = knn.BallTree(strip)
    
#     def get_neighbors(self, point, radius):
#         if self.tree.query_radius(point, r=radius, count_only=True) != 0:
#             return self.tree.query_radius(point, r=0.3)
#         return []
    
#     def remove_visited(self, points):
#         self.tree.remove(points)

class RT:
    def __init__(self, strip):
        self.strip = strip

        p = rtindex.Property()
        p.dimension = 3

        self.rt = rtindex.Index(properties=p)
        for i in range(len(strip)):
            self.rt.insert(i, self.to_index(strip[i]))
        
    def get_neighbors(self, point, radius):
        items = self.rt.intersection(self.to_index(point, radius))
        neighbors = []
        for item in items:
            point = self.strip[item]
            neighbors.append(point)
            self.rt.delete(item, self.to_index(point))
        return neighbors
    
    def to_index(self, point, radius=0):
        return (point[0] - radius, point[1] - radius, point[2] - radius, point[0] - radius, point[1] - radius, point[2] - radius)