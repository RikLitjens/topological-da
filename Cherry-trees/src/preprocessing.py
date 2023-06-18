from helpers import dist
import math
import numpy as np
from sklearn.preprocessing import normalize


def edge_evaluation(points, clusters, r_super):
    edges = []

    # Define the edges
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist(points[i], points[j]) <= 2 * r_super:
                edges.append(i, j)
                edges.append(j, i)
    
    # Convert points to new coordinate system
    for i in range(len(edges)):
        n_i = points[edges[i][0]]
        n_j = points[edges[i][1]]

        cluster_i = clusters[edges[i][0]]
        cluster_j = clusters[edges[i][1]]
        total_cluster = cluster_i.extend(cluster_j)

        # X-axis is equal to the direction of the edge
        x_axis_new = normalize(n_j - n_i)

        square_cluster = np.matmul(np.matrix.transpose(total_cluster), total_cluster)
        U, S, Vh = np.linalg.svd(square_cluster)

        min_eigenvalues = np.argsort(S)[-3:]
        min_eigenvalue_index = min_eigenvalues[0]

        # The z-axis is equal to the 3rd least significant comoponent (eigenvector belonging to the third lowest eigenvalues)
        z_axis_new = U[min_eigenvalue_index]

        # The y-axis is then perpendicular to the x_axis_new - z_axis_new plane
        y_axis_new = np.cross(z_axis_new, x_axis_new)


        # Find rotation from x, y, z to x_axis_new, y_axis_new, a_axis_new


