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
    
    def kruskal(self):
        # The edges of the final MST
        final_edges = []

        # Sort the edges by weight
        self.edges = self.edges.sort()

        uf = UnionFind()
        uf.make_set(self.vertices)

        for edge in self.edges:
            i = uf.op_find(tuple(edge.p_start))
            j = uf.op_find(tuple(edge.p_end))
            if i != j:
                final_edges.append(edge)
                UnionFind.op_union(i, j)

        result = Graph(self.point_coordinates, final_edges)
        return result

