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