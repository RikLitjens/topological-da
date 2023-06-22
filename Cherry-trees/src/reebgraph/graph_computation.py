from knn import KD

def compute_reeb(pcd, strip_size, tau):
    f = choose_f()

    point_vals = []
    for point in pcd:
        point_vals.append(point, f(point[0], point[1], point[2]))
    point_vals.sort()

    # Define the strips, i.e. the sets of points in each subdivision of the range of f. 
    # For example, range [0,1), [1,2), [2,3), etc.
    strips = []
    strip_temp = []
    min_val = point_vals[0].get_value()
    for i in range(len(point_vals)):
        if min_val + strip_size < point_vals[i].get_value():
            strips.append(strip_temp)
            strip_temp = []
            min_val = min_val + strip_size
            if min_val + strip_size < point_vals[i].get_value():
                while min_val + strip_size < point_vals[i].get_value():
                    min_val = min_val + strip_size
        else:
            strip_temp.append(point_vals[i].copy())
    
    for strip in strips:
        connected_components(strip, tau)
    
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
    return components
    



def choose_f():
    def f(x, y, z):
        return z
    return f
