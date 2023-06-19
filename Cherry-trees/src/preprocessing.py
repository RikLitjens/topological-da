from helpers import dist
import math
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image


def edge_evaluation(points, clusters, r_super):
    edges = get_edges(points, r_super)
    
    # Convert points to new coordinate system
    for k in range(len(edges)):
        n_i = points[edges[k][0]]
        n_j = points[edges[k][1]]

        total_cluster = clusters[edges[k][0]]
        total_cluster.extend(clusters[edges[k][1]])
        total_cluster = np.array(total_cluster)

        x_axis_new, y_axis_new, z_axis_new = new_coord_system(n_i, n_j, total_cluster)

        # Find rotation from x, y, z to x_axis_new, y_axis_new, a_axis_new
        # Because the axes are unit vectors (1, 0, 0), (0, 1, 0) and (0, 0, 1), 
        # the rotation matrix is equal to the new axes in order, transposed due to it being a rotation from new axes to old axes
        rotation_matrix = np.array([x_axis_new, y_axis_new, z_axis_new])

        for i in range(len(total_cluster)):
            total_cluster[i] = np.matmul(rotation_matrix, total_cluster[i])
        
        max_x = max(total_cluster[:, 0])
        max_y = max(total_cluster[:, 1])

        # Initialize the 32x16 histogram for the CNN and normalize it to 0 - 255 (image color range)
        histogram = make_greyscale(total_cluster, max_x, max_y)

        # Save the histogram to a file
        plt.imshow(histogram, interpolation='nearest', cmap='gray')
        plt.savefig(f"histogram_{k}.png")

        # img = Image.fromarray(decrypted.astype(np.uint8), 'L')
        # img.save('decrypted2.png')

def get_edges(points, r_super):
    # Define the edges
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist(points[i], points[j]) <= 2 * r_super:
                edges.append([i, j])
                edges.append([j, i])

def new_coord_system(p_o, p_t, total_cluster):
    # X-axis is equal to the direction of the edge
    x_axis_new = normalize([p_t - p_o])[0]

    square_cluster = np.matmul(np.matrix.transpose(total_cluster), total_cluster)
    U, S, Vh = np.linalg.svd(square_cluster)

    min_eigenvalues = np.argsort(S)[-3:]
    min_eigenvalue_index = min_eigenvalues[0]

    # The z-axis is equal to the 3rd least significant comoponent (eigenvector belonging to the third lowest eigenvalues)
    z_axis_n = U[min_eigenvalue_index]
    z_axis_n = normalize([z_axis_new])[0]

    # The y-axis is then perpendicular to the x_axis_new - z_axis_new plane
    y_axis_new = np.cross(z_axis_n, x_axis_new)
    y_axis_new = normalize([y_axis_new])[0]

    z_axis_new = np.cross(x_axis_new, y_axis_new)
    z_axis_new = normalize([z_axis_new])[0]

    return x_axis_new, y_axis_new, z_axis_new

def make_greyscale(total_cluster, max_x, max_y):
    histogram = np.zeros((32, 16))
    for i in range(len(total_cluster)):
        greyscale_x = total_cluster[i][0] / max_x
        greyscale_y = total_cluster[i][1] / max_y
        
        greyscale_x = int(greyscale_x * 32)
        greyscale_y = int(greyscale_y * 16)

        if greyscale_x > 31:
            greyscale_x = 31
        if greyscale_y > 15:
            greyscale_y = 15
        
        histogram[greyscale_x][greyscale_y] += 1

    # Normalize the histogram for a greyscale image
    max_hist = 255 / np.amax(histogram)
    histogram = histogram * max_hist

    return histogram