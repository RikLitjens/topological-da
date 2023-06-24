import sklearn.neighbors as knn
from rtreelib import RTree, Rect
from rtree import index as rtindex

import sys
sys.path.insert(0, fr'C:\Users\marco\Documents\GitHub\topological-da-geolife\Cherry-trees\src')

from helpers import dist

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

class KD:
    def __init__(self, strip):
        self.tree = knn.BallTree(strip)
        self.points = strip
    
    def get_neighbors(self, tuple_point, radius):
        point = [[tuple_point[0], tuple_point[1], tuple_point[2]]]
        if self.tree.query_radius(point, r=radius, count_only=True) != 0:
            ind, dist = self.tree.query_radius(point, radius, return_distance=True)
            ind = ind[0]
            result_points = []
            for i in ind:
                result_points.append(self.points[i])
            return dist, result_points
        return None, []

class RT:
    def __init__(self, strip):
        self.strip = strip

        p = rtindex.Property()
        p.dimension = 3
        p.dat_extension = 'data'
        p.idx_extension = 'index'

        self.rt = rtindex.Index('3d_index', properties=p)
        for i in range(len(strip)):
            self.rt.insert(i, self.to_index(strip[i].get_point()))
        
    def get_neighbors(self, point, radius):
        items = self.rt.intersection(self.to_index(point, radius))
        neighbors = []
        for item in items:
            new_point = self.strip[item].get_point()
            if dist(point, new_point) <= radius:
                neighbors.append(new_point)
                self.rt.delete(item, self.to_index(new_point))
        result = tuple([tuple(row) for row in neighbors])
        return result
    
    def to_index(self, point, radius=0):
        return (point[0] - radius, point[1] - radius, point[2] - radius, point[0] + radius, point[1] + radius, point[2] + radius)
    
# strip = [
#     [0,1,1],
#     [0,1.5,1],
#     [0,2.4,2.4],
#     [0,1.7,1.7],
#     [0,2,1],
#     [0,2.5,3],
#     [0.5, 1.3, 1.3],
#     [0,2.3, 2.8]
# ]

# data = RT(strip)
# print(data.get_neighbors(strip[0], 1))
# print(data.get_neighbors(strip[0], 1))
# print(data.get_neighbors(strip[2], 1))