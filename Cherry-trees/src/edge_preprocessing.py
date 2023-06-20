from helpers import dist
import math
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image
from super_points import *
from helpers import *
from deepnet.net_main import CherryLoader
import os


def edge_evaluation(edges, points, clusters, r_super, bag):    
    total_cluster = []
    # Convert points to new coordinate system
    histograms = []
    for k in range(len(edges)):
        n_i = points[edges[k][0]]
        n_j = points[edges[k][1]]

        # total_cluster = clusters[edges[k][0]]
        cluster_i = np.array(clusters[edges[k][0]])
        cluster_j = np.array(clusters[edges[k][1]])
        total_cluster = np.concatenate((clusters[edges[k][0]], clusters[edges[k][1]]), axis=0)
        # total_cluster.extend(clusters[edges[k][1]])
        # total_cluster = np.array(total_cluster)

        x_axis_new, y_axis_new, z_axis_new = new_coord_system(n_i, n_j, total_cluster)

        # Find rotation from x, y, z to x_axis_new, y_axis_new, a_axis_new
        # Because the axes are unit vectors (1, 0, 0), (0, 1, 0) and (0, 0, 1), 
        # the rotation matrix is equal to the new axes in order, transposed due to it being a rotation from new axes to old axes
        rotation_matrix = np.array([x_axis_new, y_axis_new, z_axis_new])

        for i in range(len(total_cluster)):
            total_cluster[i] = np.matmul(rotation_matrix, total_cluster[i])
        
        max_x = max(total_cluster[:, 0])
        min_x = min(total_cluster[:, 0])
        max_y = max(total_cluster[:, 1])
        min_y = min(total_cluster[:, 1])

        # Initialize the 32x16 histogram for the CNN and normalize it to 0 - 255 (image color range)
        histogram = make_greyscale(total_cluster, max_x, min_x, max_y, min_y)

        # store
        histograms.append(histogram)

        # save file
        # img = Image.fromarray(histogram.astype(np.uint8), 'L')
        # img.save(fr"Cherry-trees\images\bag{bag}histogram_{k}.png")
        # print(f"Saved image {k}")

        total_cluster = []

    return histograms

def get_edges(points, r_super):
    edges = []
    # Define the edges
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist(points[i], points[j]) <= 2 * r_super:
                edges.append([i, j])
                # edges.append([j, i])
    
    return edges

def new_coord_system(p_o, p_t, total_cluster):
    # X-axis is equal to the direction of the edge
    x_axis_new = normalize([p_t - p_o])[0]

    square_cluster = np.matmul(np.matrix.transpose(total_cluster), total_cluster)
    U, S, Vh = np.linalg.svd(square_cluster)

    min_eigenvalues = np.argsort(S)[-3:]
    min_eigenvalue_index = min_eigenvalues[0]

    # The z-axis is equal to the 3rd least significant comoponent (eigenvector belonging to the third lowest eigenvalues)
    z_axis_n = U[min_eigenvalue_index]
    z_axis_n = normalize([z_axis_n])[0]

    # The y-axis is then perpendicular to the x_axis_new - z_axis_new plane
    y_axis_new = np.cross(z_axis_n, x_axis_new)
    y_axis_new = normalize([y_axis_new])[0]

    z_axis_new = np.cross(x_axis_new, y_axis_new)
    z_axis_new = normalize([z_axis_new])[0]

    return x_axis_new, y_axis_new, z_axis_new

def make_greyscale(total_cluster, max_x, min_x, max_y, min_y):
    histogram = np.zeros((32, 16))
    for i in range(len(total_cluster)):
        greyscale_x = (total_cluster[i][0] - min_x) / (max_x - min_x)
        greyscale_y = (total_cluster[i][1] - min_y) / (max_y - min_y)
        
        if (greyscale_x == None):
            print(f"This x_position is {total_cluster[i][0]} but results in NaN")
        if (greyscale_y == None):
            print(f"This y_position is {total_cluster[i][1]} but results in NaN")
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

def test_image_creation():
    # path = fr"Cherry-trees\images\Training\bag0histogram_0.png"
    ROOT_DIR = os.path.abspath(os.curdir)
    files = os.listdir(os.path.join(ROOT_DIR, "Cherry-trees", "images", "Training"))
    files.sort()

    for i in range(len(files)):
        bag = max(0, math.floor(i / 200))
        number = i - bag * 200
        histogram = get_image(fr"Cherry-trees\images\Training\{files[i]}")
        histograms = CherryLoader().explode_data(histogram, 3)
        for i in range(len(histograms)):
            img = Image.fromarray(histograms[i].astype(np.uint8), 'L')
            img.save(fr"Cherry-trees\images\exploded\bag{bag}histogram{number}resample{i}.png")
