class Skeleton:
    def __init__(self) -> None:
        self.edges = []


    def get_skel_score(self):
        return sum([self.get_reward(edge, None) for edge in self.edges])
    
    def get_reward(self, edge, label):
        edge_score = self.get_edge_score(edge, 0)
        turn_penalty = self.get_turn_penalty(edge)
        grow_penalty = 0
        return edge_score - turn_penalty - grow_penalty
    
    def get_edge_score(self, edge, alpha):
        edge.length * (1 - ((1-edge.conf) / (1-alpha)))

    def get_turn_penalty(self, edge, theta, c_turn, p_turn):
        if edge.label is not None and edge.label != edge.pre.label:
            return 0
        
        if edge.angle_with(edge.pre) <= edge.pre:
            return 0
        
        return c_turn * (edge.angle_with(edge.pre) - theta) ** p_turn
    
    def get_growth_penalty(self, edge):
        pass

