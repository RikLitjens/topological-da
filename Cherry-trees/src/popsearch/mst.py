from edge import Edge
import numpy as np
import math

class UnionFind:
    parent_node = {}
    rank = {}

    def make_set(self, u):
        for i in u:
            self.parent_node[i] = i
            self.rank[i] = 0

    def op_find(self, k):
        if self.parent_node[k] != k:
            self.parent_node[k] = self.op_find(self.parent_node[k])
        return self.parent_node[k]

    def op_union(self, a, b):
        x = self.op_find(a)
        y = self.op_find(b)

        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent_node[y] = x
        elif self.rank[x] < self.rank[y]:
            self.parent_node[x] = y
        else:
            self.parent_node[x] = y
            self.rank[y] = self.rank[y] + 1

class Graph:
    def __init__(self, points, edges):
        self.vertices = []
        for point in points:
            self.vertices.append(tuple(point))
        self.edges = edges
    
    def heighest(self):
        heighest = 0
        for vertex in self.vertices:
            heighest = max(heighest, vertex[2])
        return heighest
    
    def lowest(self):
        lowest = math.inf
        for vertex in self.vertices:
            lowest = min(lowest, vertex[2])
        return lowest
    
    def kruskal(self):
        # The edges of the final MST
        final_edges = []

        # Sort the edges by weight
        self.edges.sort()

        uf = UnionFind()
        uf.make_set(self.vertices)

        for edge in self.edges:
            i = uf.op_find(tuple(edge.p_start))
            j = uf.op_find(tuple(edge.p_end))
            if i != j:
                final_edges.append(edge)
                uf.op_union(i, j)

        result = Graph(self.vertices, final_edges)
        return result
    
def generate_tree(points, edges, alpha_tip):
    graph = Graph(points, edges)
    mst = graph.kruskal()

    max_height = graph.heighest()
    min_height = graph.lowest()
    final_mst = []

    for edge in mst.edges:
        if edge.calculate_grow_angle() < np.pi / 4:
            final_mst.append(edge)
            continue
        threshold = max_height - alpha_tip * (max_height - min_height)
        if (edge.p_start[2] < threshold) or (edge.p_end[2] < threshold):
            final_mst.append(edge)
    
    return Graph(points, final_mst)



# This is an example for testing the MST!

# vertices = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, -2, 0], [4, 0, 0]]
# edg = [Edge(vertices[0], vertices[1], 0, None),
#        Edge(vertices[0], vertices[2], 0, None),
#        Edge(vertices[1], vertices[2], 0, None),
#        Edge(vertices[2], vertices[3], 0, None),
#        Edge(vertices[3], vertices[4], 0, None),
#        Edge(vertices[2], vertices[4], 0, None)]

# graph = Graph(vertices, edg)
# print(graph.vertices)
# print(graph.edges)
# mst_edges = graph.kruskal()
# for edge in mst_edges.edges:
#     print(f"Edge from {edge.p_start} to {edge.p_end}")