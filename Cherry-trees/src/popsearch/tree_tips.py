from popsearch.skeleton_components import Edge
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from popsearch.graph import Graph


def cut_tree(points, edges, alpha_tip):
    graph = Graph(points, edges)
    mst = graph.kruskal()

    max_height = graph.heighest()
    min_height = graph.lowest()
    final_mst = []

    for edge in mst.edges:
        if edge.calculate_grow_angle() < np.pi / 4:
            continue
        threshold = max_height - alpha_tip * (max_height - min_height)
        if (edge.p1[2] < threshold) or (edge.p2[2] < threshold):
            continue

        final_mst.append(edge)

    return Graph(points, final_mst), threshold


# This is an example for testing the MST!


def test_mst():
    vertices = [[0, 0, 0], [0, 2, 3], [0, 1, 5], [0, -1, 7], [0, 10, 6], [0, 10, 8]]
    edg = [
        Edge(vertices[0], vertices[1], 0, None),
        Edge(vertices[0], vertices[2], 0, None),
        Edge(vertices[1], vertices[2], 0, None),
        Edge(vertices[2], vertices[3], 0, None),
        Edge(vertices[1], vertices[4], 0, None),
        Edge(vertices[4], vertices[5], 0, None),
    ]

    graph = Graph(vertices, edg)

    mst_edges = graph.kruskal()
    cut, threshold = cut_tree(vertices, edg, 0.4)

    # print(graph.vertices)
    # print(graph.edges)
    # for edge in mst_edges.edges:
    #     print(f"Edge from {edge.p1} to {edge.p2}")

    print(len(graph.find_connected_components()))
    print(len(cut.find_connected_components()))

    cut_tips = cut.find_tree_tips()
    print("cut")
    print(cut_tips)
    print(cut.find_connected_components())

    # plot
    graph.plot()
    mst_edges.plot()

    cut.plot()
    cut.plot(cut_tips, threshold)
