class PointVal:
    def __init__(self, point, value):
        self.point = point
        self.value = value
    
    def get_value(self):
        return self.value
    
    def get_point(self):
        return self.point
    
    def __lt__(self, other):
         return self.get_value() < other.get_value()
    
    def __eq__(self, other: object) -> bool:
        return self.point == other.point and self.value == other.value
    
class ReebNode:
    def __init__(self, point, point_cloud, interval):
        self.point = point
        self.point_cloud = point_cloud
        self.interval = interval
        self.adj = {}
    
    def get_pointcloud(self):
        return self.point_cloud

    def add_edge(self, other):
        self.adj.add(other)