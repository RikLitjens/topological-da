from helpers import numpy_to_pcd, visualize_point_cloud
from reebgraph.knn import KD, RT
from reebgraph.reeb_graph import PointVal, ReebNode

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
        visualize_point_cloud(numpy_to_pcd(vis_list))

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
    data = RT(strip)

    points = set()
    for point in strip:
        coord = point.get_point()
        points.add((coord[0], coord[1], coord[2]))

    components = []
    while len(points) > 0:
        new_point = points.pop()
        component = set()

        to_visit = {new_point}
        while len(to_visit) > 0:
            to_visit.remove(new_point)
            component.add(new_point)

            neighbors = data.get_neighbors(new_point, tau)

            neighbors = set(neighbors)
            points.remove(neighbors)

            to_visit.add(neighbors)
            new_point = neighbors.pop()
        points.remove(component)
        components.append(component)
    return components, compute_centroids(components)

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

def tau_connected(pcd1, pcd2, tau):
    pos_tree = KD(pcd2)
    
    for point in pcd1:
        dist, throwaway = pos_tree.get_neighbours(point, tau)
        if len(throwaway) > 0:
            return True
    return False

def find_reeb_nodes(strips, ranges, tau):
    # Find each component's centroid and create a ReebNode for it. 
    # Using ranges, make sure to store the range of f that the node is part of. 
    reeb_nodes = []
    for i in range(len(strips)):
        components, centroids = connected_components(strips[i], tau)
        temp_nodes = []
        for i in range(len(components)):
            temp_nodes.append(ReebNode(centroids[i], components[i], ranges[i]))
        reeb_nodes.append(temp_nodes)
    
    get_edges(reeb_nodes, tau)
    return reeb_nodes

def get_edges(reeb_nodes, tau):
    for strip in reeb_nodes:
        for i in range(len(strip)):
            for j in range(len(strip)):
                if i != j:
                    node1 = strip[i]
                    node2 = strip[j]
                    if tau_connected(node1.get_pointcloud(), node2.get_pointcloud(), tau):
                        strip[i].add_edge(strip[j])
                        strip[j].add_edge(strip[i])

def choose_f():
    def f(x, y, z):
        return z
    return f