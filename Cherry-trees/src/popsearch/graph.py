import matplotlib.pyplot as plt
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
            i = uf.op_find(tuple(edge.p1))
            j = uf.op_find(tuple(edge.p2))
            if i != j:
                final_edges.append(edge)
                uf.op_union(i, j)

        result = Graph(self.vertices, final_edges)
        return result

    def plot(self, tips=[], threshold=None):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the vertices
        for vertex in self.vertices:
            x, y, z = vertex
            c = "r" if vertex not in tips else "g"
            ax.scatter(x, y, z, c=c, marker="o")

        # Plot the MST edges
        for edge in self.edges:
            x_coords = [edge.p1[0], edge.p2[0]]
            y_coords = [edge.p1[1], edge.p2[1]]
            z_coords = [edge.p1[2], edge.p2[2]]
            ax.plot(x_coords, y_coords, z_coords, c="b")

        # Set labels and display the plot
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def find_connected_components(self, with_edges=True):
        representation = {tuple(vertex): [] for vertex in self.vertices}
        visited = set()
        components = []

        for edge in self.edges:
            representation[tuple(edge.p1)].append(tuple(edge.p2))
            representation[tuple(edge.p2)].append(tuple(edge.p1))

        def dfs(node, component):
            visited.add(tuple(node))
            component.append(tuple(node))

            for neighbor in representation[tuple(node)]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for vertex in self.vertices:
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(component)

        if with_edges:
            edge_components = []
            for component in components:
                if len(component) == 1:
                    continue
                edge_components.append(component)
            return edge_components

        return components

    def find_tree_tips(self):
        components = self.find_connected_components(True)
        tips = []
        for component in components:
            tip = sorted(component, key=lambda x: x[2], reverse=True)[0]
            tips.append(tip)
        return tips
