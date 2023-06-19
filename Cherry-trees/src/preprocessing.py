from helpers import dist
import math
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image


def edge_evaluation(points, clusters, r_super):
    edges = []

    # Define the edges
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist(points[i], points[j]) <= 2 * r_super:
                edges.append(i, j)
                edges.append(j, i)
    
    # Convert points to new coordinate system
    for k in range(len(edges)):
        n_i = points[edges[k][0]]
        n_j = points[edges[k][1]]

        cluster_i = clusters[edges[k][0]]
        cluster_j = clusters[edges[k][1]]
        total_cluster = cluster_k.extend(cluster_j)

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
        # Because the axes are unit vectors (1, 0, 0), (0, 1, 0) and (0, 0, 1), 
        # the rotation matrix is equal to the new axes in order, transposed due to it being a rotation from new axes to old axes
        rotation_matrix = np.matrix.transpose(np.array([x_axis_new, y_axis_new, z_axis_new]))

        for i in range(len(total_cluster)):
            total_cluster[i] = np.matmul(rotation_matrix, total_cluster[i])
        
        max_x = max(total_cluster[:, 0])
        max_y = max(total_cluster[:, 1])

        # Initialize the 32x16 histogram for the CNN
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

        # Save the histogram to a file
        plt.imshow(histogram, interpolation='nearest', cmap='gray')
        plt.savefig(f"histogram_{k}.png")

        # img = Image.fromarray(decrypted.astype(np.uint8), 'L')
        # img.save('decrypted2.png')
