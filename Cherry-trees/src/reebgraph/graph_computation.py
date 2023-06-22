

def compute_reeb(pcd, strip_size):
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
                    strips.append([])
                    min_val = min_val + strip_size
        strip_temp.append(point_vals[i].copy())
    
def connected_components(strip):
    # Find the connected components in a strip
    # Returns a list of lists of points


def choose_f():
    def f(x, y, z):
        return z
    return f
