from matplotlib import pyplot as plt
from helpers import choose_f, dist, get_data_path, load_point_cloud, numpy_to_pcd, visualize_point_cloud, filter_data
from reebgraph.knn import KD, RT
from reebgraph.reeb_graph import PointVal, ReebNode
from sklearn.cluster import DBSCAN
import numpy as np
from preprocess import rotate_z_up

def compute_reeb(pcd, strip_size, tau):
    f = choose_f()

    point_vals = []
    for point in pcd:
        point_vals.append(PointVal(point, f(point[0], point[1], point[2])))
    point_vals.sort()
    
    strips, ranges = get_strips(point_vals, strip_size)

    strip_pcds = []
    for i in range(len(strips)):
        vis_list = []
        for point_val in strips[i]:
            vis_list.append(point_val.get_point())
        strip_pcd = np.asarray(vis_list)
        strip_pcd = numpy_to_pcd(vis_list)
        strip_pcd.paint_uniform_color([i % 2, 0, (i + 1) % 2])
        strip_pcds.append(strip_pcd)
    # visualize_point_cloud(strip_pcds)

    reeb_nodes = find_reeb_nodes(strips, ranges, tau)
    
    return reeb_nodes
    
def get_strips(point_vals, strip_size):
    # Define the strips, i.e. the sets of points in each subdivision of the range of f. 
    # For example, range [0,1), [1,2), [2,3), etc.
    strips = []
    ranges = []
    strip_temp = []
    min_val = point_vals[0].get_value()
    for i in range(len(point_vals)):
        if min_val + strip_size < point_vals[i].get_value():
            strips.append(strip_temp)
            ranges.append((min_val, min_val + strip_size))
            strip_temp = []
            min_val = min_val + strip_size
            if min_val + strip_size < point_vals[i].get_value():
                while min_val + strip_size < point_vals[i].get_value():
                    min_val = min_val + strip_size
        else:
            strip_temp.append(PointVal(point_vals[i].get_point(), point_vals[i].value))
    
    return strips, ranges

def connected_components(strip, tau):
    # Find the connected components in a strip
    # Returns a list of lists of points
    # data = RT(strip)

    points = []
    for point in strip:
        coord = point.get_point()
        points.append(coord)

    # components = []
    # while len(points) > 0:
    #     component = set()

    #     next_point = points.pop()
    #     to_visit = {next_point}
    #     while len(to_visit) > 0:
    #         new_point = to_visit.pop()
    #         component.add(new_point)

    #         neighbors = data.get_neighbors(new_point, tau)

    #         for neighbor in neighbors:
    #             points.discard(neighbor)
    #             to_visit.add(neighbor)
    #     points -= component
    #     components.append(component)

    points = np.array(points)
    components = DBSCAN(eps=tau, min_samples=23, metric='euclidean').fit(points).labels_
    j = 0
    all_components = [[]]
    indices = {0 : 0}
    for i in range(len(points)):
        if components[i] >= 0:
            if not (components[i] in indices.keys()):
                indices[components[i]] = len(all_components)
                all_components.append([])
            all_components[indices[components[i]]].append((points[i][0], points[i][1], points[i][2]))
    final_components = []
    for component in all_components:
        if len(component) > 5:
            final_components.append(component)
    
    return final_components, compute_centroids(final_components)

def compute_centroids(components):
    # Compute the centroids of the connected components
    centroids = []

    for component in components:
        x = 0
        y = 0
        z = 0
        for point in component:
            x = x + point[0]
            y = y + point[1]
            z = z + point[2]
        x = x / len(component)
        y = y / len(component)
        z = z / len(component)
        centroids.append((x, y, z))
    return centroids

def tau_connected(pcd1, kd2, tau):
    for point in pcd1:
        dist = kd2.closest_neighbor(point)
        if dist <= 2*tau:
            return True
    return False

def find_reeb_nodes(strips, ranges, tau):
    # Find each component's centroid and create a ReebNode for it. 
    # Using ranges, make sure to store the range of f that the node is part of. 
    reeb_nodes = []
    print(f"This is the number of strips {len(strips)} and this is the number of ranges {len(ranges)}")
    for i in range(len(strips)):
        components, centroids = connected_components(strips[i], tau)
        temp_nodes = []
        print(f"This is the number of components {len(components)} and this is the number of centroids {len(centroids)}")
        for j in range(len(components)):
            temp_nodes.append(ReebNode(centroids[j], components[j], ranges[i]))
        reeb_nodes.append(temp_nodes)
    
    ps_total = []
    reeb_total = []
    for subnode in reeb_nodes:
        reeb_points = []
        ps_row = []
        for node in subnode:
            reeb_point = [node.get_point()[0], node.get_point()[1], node.get_point()[2]]
            reeb_points.append(reeb_point)
            reeb_total.append(reeb_point)

            ps_fake = node.get_pointcloud()
            ps = []
            for (x, y, z) in ps_fake:
                ps.append([x, y, z])
                ps_row.append([x, y, z])
                ps_total.append([x, y, z])

    reeb_total = np.asarray(reeb_total)
    reeb_total_pcd = numpy_to_pcd(reeb_total)
    # visualize_point_cloud([reeb_total_pcd])
    get_edges(reeb_nodes, tau)

    reeb_graph = []
    for group in reeb_nodes:
        for node in group:
            reeb_graph.append(node)

    return reeb_graph

def get_edges(reeb_nodes, tau):
    for i in range(len(reeb_nodes) - 1):
        print(f"This is arc-computation iteration {i}")
        j = i + 1

        kd_j = []
        for node in reeb_nodes[j]:
            kd_j.append(KD(node.get_pointcloud()))

        for node in reeb_nodes[i]:
            for k in range(len(kd_j)):
                kd = kd_j[k]
                print(f"In this iteration, node1 has {len(node.get_pointcloud())} points")
                node_dist = dist(node.get_point(), reeb_nodes[j][k].get_point())
                if node_dist <= 2*tau + node.get_convex_size() + reeb_nodes[j][k].get_convex_size():
                    if tau_connected(node.get_pointcloud(), kd, tau):
                        node.add_edge(reeb_nodes[j][k])
                        reeb_nodes[j][k].add_edge(node)

def plot_reeb(reeb_nodes):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the vertices
        vertices = []
        for vertex in reeb_nodes:
            x, y, z = vertex.get_point()
            vertices.append([x, y, z])
            # ax.scatter(x, y, z, c='r', marker="o")
            for edge in vertex.adj:
                x_coords = [vertex.get_point()[0], edge.get_point()[0]]
                y_coords = [vertex.get_point()[1], edge.get_point()[1]]
                z_coords = [vertex.get_point()[2], edge.get_point()[2]]
                ax.plot(x_coords, y_coords, z_coords, c="black")
        vertices = np.asarray(vertices)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', marker="o", s=2)

        # Set labels and display the plot
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Top view")
        ax.view_init(elev=90, azim=-90)
        plt.savefig(f'figures/reeb_top.png', bbox_inches='tight', dpi=300)

        ax.set_title("Front view")
        ax.view_init(elev=0, azim=-90)
        plt.savefig(f'figures/reeb_front.png', bbox_inches='tight',dpi=300)

        ax.set_title("Side view")
        ax.view_init(elev=0, azim=0)
        plt.savefig(f'figures/reeb_side.png', bbox_inches='tight',dpi=300)