from helpers import choose_f, dist


class PointVal:
    def __init__(self, point, value):
        self.coord = point
        self.value = value
    
    def get_value(self):
        return self.value
    
    def get_point(self):
        return self.coord
    
    def __lt__(self, other):
         return self.get_value() < other.get_value()
    
    def __eq__(self, other: object) -> bool:
        return self.coord == other.coord and self.value == other.value
    
    def copy(self):
        return PointVal(self.coord, self.value)
    
class ReebNode:
    def __init__(self, point, point_cloud, interval):
        self.point = point
        self.interval = interval
        self.adj = set()
        self.f = choose_f()

        self.point_cloud = point_cloud
        max_dist = 0
        for point in point_cloud:
            max_dist = max(max_dist, dist([point[0], point[1], point[2]], self.point))
        self.size = max_dist

        # point_vals = []
        # max_dist = 0
        # for point in point_cloud:
        #     point_vals.append(PointVal(point, self.f(point[0], point[1], point[2])))
        #     max_dist = max(max_dist, dist([point[0], point[1], point[2]], self.point))
        # self.size = max_dist
        # point_vals.sort()
        # self.point_cloud = []
        # max_ind = len(point_vals) - 1
        # for i in range(len(point_vals)):
        #     self.point_cloud.append(point_vals[max_ind - i].get_point())
    
    def get_pointcloud(self):
        return self.point_cloud

    def get_convex_size(self):
        return self.size
    
    def get_point(self):
        return self.point

    def add_edge(self, other):
        self.adj.add(other)