from knn import KD
from reeb_graph import ReebNode

def compute_reeb(pcd, strip_size, tau):
    f = choose_f()

    point_vals = []
    for point in pcd:
        point_vals.append(point, f(point[0], point[1], point[2]))
    point_vals.sort()

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
            strip_temp.append(point_vals[i].copy())
    

    reeb_nodes = []
    for i in range(len(strips)):
        components, centroids = connected_components(strips[i], tau)
        temp_nodes = []
        for i in range(len(components)):
            temp_nodes.append(ReebNode(centroids[i], components[i], ranges[i]))
        reeb_nodes.append(temp_nodes)
    
    for strip in reeb_nodes:
        for i in range(len(strip)):
            for j in range(len(strip)):
                if i != j:
                    node1 = strip[i]
                    node2 = strip[j]
                    if tau_connected(node1.get_pointcloud(), node2.get_pointcloud(), tau):
                        strip[i].add_edge(strip[j])
                        strip[j].add_edge(strip[i])
    
    return reeb_nodes
    
def connected_components(strip, tau):
    # Find the connected components in a strip
    # Returns a list of lists of points
    data = KD(strip)

    points = {}
    for point in strip:
        points.add((point[0], point[1], point[2]))

    components = []
    while len(points) > 0:
        new_point = points.pop()
        component = {}

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


def choose_f():
    def f(x, y, z):
        return z
    return f
