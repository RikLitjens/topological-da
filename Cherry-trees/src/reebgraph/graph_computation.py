from helpers import choose_f, dist, get_data_path, load_point_cloud, numpy_to_pcd, visualize_point_cloud
from reebgraph.knn import KD, RT
from reebgraph.reeb_graph import PointVal, ReebNode
from sklearn.cluster import DBSCAN
import open3d as o3d
import numpy as np

def compute_reeb(pcd, strip_size, tau):
    f = choose_f()

    point_vals = []
    for point in pcd:
        point_vals.append(PointVal(point, f(point[0], point[1], point[2])))
    point_vals.sort()
    
    strips, ranges = get_strips(point_vals, strip_size)

    for strip in strips:
        vis_list = []
        for point_val in strip:
            vis_list.append(point_val.get_point())
        # visualize_point_cloud(numpy_to_pcd(vis_list))

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

    components = DBSCAN(eps=tau, algorithm='ball_tree').fit(points).labels_
    print(components)
    j = 0
    all_components = [[]]
    for i in range(len(points)):
        if components[i] > j:
            all_components.append([])
            j = j + 1
        all_components[j].append((points[i][0], points[i][1], points[i][2]))
    final_components = []
    for component in all_components:
        if len(component) > 5:
            final_components.append(component)
    
    print(f"I got here! I found {len(final_components)} components.")
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
        if dist <= tau:
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
    
    reeb_points = []
    for subnode in reeb_nodes:
        for node in subnode:
            reeb_point = [node.get_point()[0], node.get_point()[1], node.get_point()[2]]
            reeb_points.append(reeb_point)

            ps_fake = node.get_pointcloud()
            ps = []
            for (x, y, z) in ps_fake:
                ps.append([x, y, z])
            ps = np.asarray(ps)
            ps = numpy_to_pcd(ps)
            ps.paint_uniform_color([1, 0, 0])
            reeb_point = np.asarray([reeb_point])
            reeb_node = numpy_to_pcd(reeb_point)
            reeb_node.paint_uniform_color([0, 1, 0])
            visualize_point_cloud([ps, reeb_node])
    
    get_edges(reeb_nodes, tau)
    return reeb_nodes

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
                if node_dist <= tau + node.get_convex_size() + reeb_nodes[j][k].get_convex_size():
                    if tau_connected(node.get_pointcloud(), kd, tau):
                        node.add_edge(reeb_nodes[j][k])
                        reeb_nodes[j][k].add_edge(node)

    # for strip in reeb_nodes:
    #     print("Another strip visited")
    #     for i in range(len(strip)):
    #         for j in range(len(strip)):
    #             if i != j:
    #                 node1 = strip[i]
    #                 node2 = strip[j]
    #                 if tau_connected(node1.get_pointcloud(), node2.get_pointcloud(), tau):
    #                     strip[i].add_edge(strip[j])
    #                     strip[j].add_edge(strip[i])