from enum import Enum
from queue import PriorityQueue
import numpy as np
import heapq

class LabelEnum(Enum):
    TRUNK = 0
    SUPPORT = 1
    LEADER = 2
    SIDE = 3

class Edge:
    def __init__(self, p_start, p_end, conf=0, label=None):
        self.p_start = p_start
        self.p_end = p_end
        self.label = label
        self.conf = conf

        # not yet known at start
        self.dijk = (None, []) # list with tip n and all edges to it from self
        self.proper_pre = None

    def angle_with(self, edge2):
        # Calculate the vectors representing the edges
        vector1 = np.array(self.p_end) - np.array(self.p_start)
        vector2 = np.array(edge2.p_end) - np.array(edge2.p_start)
        
        # Calculate the dot product of the vectors
        dot_product = np.dot(vector1, vector2)
        
        # Calculate the magnitudes of the vectors
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        # Calculate the angle in radians
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
        
        return angle_radians

    def calculate_grow_angle(self):
        # Define the vector representing the edge
        edge_vector = np.array(self.p_end) - np.array(self.p_start)
        
        # Calculate the dot product of the edge vector with the XZ plane vectors
        dot_product_z = np.dot(edge_vector, np.array([0, 0, 1]))
        dot_product_x = np.dot(edge_vector, np.array([1, 0, 0]))
        
        # Calculate the absolute values of the dot products
        abs_dot_product_z = np.abs(dot_product_z)
        abs_dot_product_x = np.abs(dot_product_x)
        
        # Calculate the angle in radians
        angle_radians = np.arctan(abs_dot_product_z / abs_dot_product_x)
        
        return angle_radians
    
    def length(self):
        # Calculate the vector representing the edge
        edge_vector = np.array(self.p_end) - np.array(self.p_start)
        
        # Calculate the length of the edge using the Euclidean distance formula
        edge_length = np.linalg.norm(edge_vector)
        
        return edge_length
    
    def get_reward(self):
        """ 
        Get the optimizer score of this edge
        Based on the confidence value
        the turn penalty
        and the grow direction
        """
        edge_score = self.get_edge_score(0)
        turn_penalty = self.get_turn_penalty(self.proper_pre, 0,0,0)
        growth_penalty = self.get_growth_penalty(0,0,0)

        return edge_score - turn_penalty - growth_penalty
    
    def get_edge_score(self, alpha):
        return self.length() * (1 - ((1 - self.conf) / (1 - alpha)))
    
    def get_turn_penalty(self, pre, theta, c_turn, p_turn):
        """
        Turning edges are not wanted,
        straight is better (no offence to the lgbt)
        """
        if self.label is not None and self.label != pre.label:
            return 0

        if self.angle_with(pre) <= theta:
            return 0

        return c_turn * (self.angle_with(pre) - theta) ** p_turn
    
    def get_growth_penalty(self, theta, c_grow, p_grow):
        """"
        Returns a penalty when the label and grow direction do not match
        """
        if self.label in [LabelEnum.LEADER, LabelEnum.SUPPORT]:
            return 0
        if self.get_delta_target() <= theta:
            return 0
        
        return c_grow * (self.get_delta_target() - theta) ** p_grow
    
    def get_delta_target(self, edge):
        """
        Helper function that returns the angle 
        of an edge to its label
        i.e. support has to be horizontal
        and leader vertical
        """
        grow_angle = edge.angle_with_x()
        if edge.label == LabelEnum.SUPPORT:
            return grow_angle

        if edge.label == LabelEnum.LEADER:
            return np.pi / 2 - grow_angle
        
    def get_dijkstra_weight(self):
        # cannot have a label in this step, so
        # remove to None
        label = self.label
        self.label = None 
        score =  self.length() * (1 - self.conf) + self.get_turn_penalty(0,0,0)
        self.label = label
        return score

    
class Skeleton:
    def __init__(self, superpoints, raw_edges, tree_tips) -> None:
        # define suerpoints and raw edges (not added yet but could be)
        self.superpoints = superpoints
        self.raw_edges = raw_edges

        # add dynamically
        self.proper_edges = []
        self.tree_tips = tree_tips
        self.base_node = tree_tips[0] #TODO TODO TODO

        # create a map from superpoint to index
        self.superpoints_idx_map = {}
        for index, item in enumerate(self.superpoints):
            self.superpoints_idx_map[item] = index

        # create a map to track end points of edges
        # and maps them to the edge itself
        # for connectedness and 1 predecessor rule
        # KEYS: endpoints of predecessor
        #  used when adding an edge with p_start == p_end of predecessor
        self.predecessor_map = {point: None for point in self.superpoints}

        # this can be a list
        self.successor_map = {point: [] for point in self.superpoints}

        # map to get raw edges
        self.raw_edge_map = {(edge.p_start, edge.p_end): edge for edge in self.raw_edges}
        

    # map to check connectedness and 
    def add_eligible_edge(self, edge):
        self.proper_edges.append(edge)
        self.predecessor_map[edge.p_end] = edge
        self.successor_map[edge.p_start].append(edge)

        # add predecessor after adding
        edge.pre = self.predecessor_map[edge.p_start]


    def remove_last_proper_edge(self):
        """
        Throws out the last added edge
        """
        last_edge = self.proper_edges[-1]
        # throw it out
        self.proper_edges = self.proper_edges[:-1]

        # throw out of predecessor maps
        self.predecessor_map[last_edge.p_end] = None
        self.successor_map[last_edge.p_start] = self.successor_map[last_edge.p_start][:-1]
        last_edge.pre = None

    def get_skel_score(self):
        """
        Skel score is the optimization goal value
        """
        return sum([self.get_reward(edge, None) for edge in self.proper_edges])
    
    def get_skel_potential_after(self, eligible_edge):
        # add edge and remove after
        self.add_eligible_edge(eligible_edge)

        # get first two potential components
        new_skel_score = self.get_skel_score()
        dijk_edge_score = sum([ed.get_edge_score() for ed in eligible_edge.dijk[1]])

        # last one requires more work
        dijk_turn_penalties = []
        for i, ed in enumerate(eligible_edge.dijk[1]):
            next_edge = eligible_edge.dijk[1][i+1]
            dijk_turn_penalties.append(ed.get_turn_penalty(next_edge, 0, 0, 0))

        # Sort the list in descending order
        sorted_penalties = sorted(dijk_turn_penalties, reverse=True)

        # Drop the highest values
        n_dropped = 2 - eligible_edge.label
        dropped_penalties = sorted_penalties[n_dropped:]

        # calculate final potantial component
        dijk_turn_penalty_score = sum(dropped_penalties)

        # remove the added edge again, ONLY POTENTIAL
        self.remove_last_proper_edge()

        return new_skel_score + dijk_edge_score - dijk_turn_penalty_score


    def get_eligible(self, n_tip):
        # initialize dijkstra
        weights = [None for _ in range(len(self.superpoints)) for _ in range(len(self.superpoints))]
        for edge in self.raw_edges:
            start_index = self.superpoints_idx_map[edge.p_start]
            end_index = self.superpoints_idx_map[edge.p_end]
            # undirected
            weights[start_index][end_index]  =  edge.get_dijkstra_weight()    
            weights[end_index][start_index]  =  edge.get_dijkstra_weight()   
        

        eligible_edges = []
        for raw_edge in self.raw_edges:
            # elegibile is based on dijkstra
            min_distance, dijk = self.dijkstra(weights, self.raw_edge_map, n_tip, raw_edge)
            raw_edge.dijk = (n_tip, dijk) # save for possible later use
            eligible = True

            # check for first property of path existence
            if min_distance == float('inf'):
                eligible = False
            
            # check for basic topology violations
            if self.violate_basic_topology(raw_edge):
                eligible = False

            # and label violation
            if self.violate_label_topology(raw_edge):
                eligible = False

            # if elegible add to the list
            if eligible: eligible_edges.append(raw_edge)
            
        return eligible_edges

    
    def dijkstra(self, graph_weights, edge_map, start, candidate_edge):
        start_idx = self.superpoints_idx_map[start]
        v = len(self.superpoints)
        visited = []
        D = {v:float('inf') for v in range(v)}
        predecessors = {v: None for v in range(v)}  # Track predecessors
        D[start] = 0
        

        q = PriorityQueue()
        q.put((0, start))

        while not q.empty():
            (dist, current_point_idx) = q.get()
            visited.append(current_point_idx)

            for neighbor_idx in range(v):
                if graph_weights[current_point_idx][neighbor_idx] is not None:
                    distance = graph_weights[current_point_idx][neighbor_idx]
                    if neighbor_idx not in visited:
                        old_cost = D[neighbor_idx]
                        new_cost = D[current_point_idx] + distance
                        if new_cost < old_cost:
                            q.put((new_cost, neighbor_idx))
                            D[neighbor_idx] = new_cost
                            predecessors[neighbor_idx] = current_point_idx

        # shortest path to edge endpoint 1
        if D[self.superpoints_idx_map[candidate_edge.p_start]] > D[self.superpoints_idx_map[candidate_edge.p_end]]:
            # loop through predecessors to find all edges
            current_pre = candidate_edge.p_end
            min_dist = D[self.superpoints_idx_map[candidate_edge.p_end]]
        
        # shortest path to edge endpoint 2
        else:
            current_pre = candidate_edge.p_start
            min_dist =  D[self.superpoints_idx_map[candidate_edge.p_start]]

        # get predecessors (Dijk in paper
        dijk = []
        
        while predecessors[current_pre] is not None:
            # add edge that belongs to it
            predecessor_edge = self.raw_edge_map[(current_pre, predecessors[current_pre])]
            dijk.append(predecessor_edge)
            current_pre = predecessors[current_pre]

        return min_dist, dijk

        
    def violates_basic_topology(self, candidate_edge):

        # if the endpoint of the edge is already the endpoint of another edge, 
        # this is wrong and should be deleted
        # by the principle of acylic outward graph
        if self.predecessor_map[candidate_edge.p_end] > 0:
            return True
        
        # the starting point of the edge has to coincide
        # with the end point of another
        # otherwise the graph is not connected
        if self.predecessor_map[candidate_edge.p_start] == 0 and not candidate_edge.p_start == self.base_node:
            return True
        
        return False
    
    def violate_label_topology(self, candidate_edge):
        predecessor =  self.predecessor_map[candidate_edge.p_start]
        successors = self.successor_map[candidate_edge.p_start]

        # label progression
        if candidate_edge.label < predecessor.label:
            return True


        # label linearity
        successors_candidate = successors + [candidate_edge]
        if len(successors_candidate) > 1:
            all_equal = True
            prev = successors_candidate[0]
            for succ in successors_candidate[1:]:
                 if succ.label != prev.label:
                     all_equal = False

            if all_equal: return True

        # Trunk_support split, check if relevant
        if len(successors) > 0 and predecessor.label == LabelEnum.TRUNK:
            # check if successors are all trunks
            # check if none of them are trunks
            # count the support succs
            all_trunk = True
            no_trunk = True
            support_count = 0
            for succ in successors:
                if succ.label != LabelEnum.TRUNK:
                    all_trunk = False
                else:
                    no_trunk = False

                if succ.label == LabelEnum.SUPPORT:
                    support_count += 1

            if not all_trunk and not no_trunk:
                return True
            
            if support_count > 2:
                return True
                
        # otherwise
        return False

                





    

        


        

