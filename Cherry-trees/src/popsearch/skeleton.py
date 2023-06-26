from popsearch.edge import EdgeCollection, LabelEnum

from queue import PriorityQueue
import numpy as np



class Skeleton:
    def __init__(self, superpoints, all_edges, base_node) -> None:
        # define suerpoints and raw edges (not added yet but could be)
        self.superpoints = [tuple(p) for p in superpoints]
        self.base_node = base_node
        self.all_edge_collection = EdgeCollection(self.superpoints, all_edges)
        self.included_edge_collection = EdgeCollection(self.superpoints, []) 

        # create a map from superpoint to index in the list
        self.superpoints_idx_map = {}
        for index, item in enumerate(self.superpoints):
            self.superpoints_idx_map[item] = index


    def include_eligible_edge(self, edge):
        """
        Adds an edge to the final skeleton
        """
        self.included_edge_collection.add_edge(edge)


    def exclude_last_included_edge(self):
        """
        Throws out the last added edge
        """
        self.included_edge_collection.remove_last_edge()

    def get_skel_score(self):
        """
        Skel score is the optimization goal value
        """
        return sum([self.get_reward(edge, self.included_edge_collection.get_unique_predecessor(edge), self.base_node) for edge in self.included_edge_collection.edges])
    
    def get_potential(self, eligible_edge):
        """
        Calculate the edge potential as in the paper
        """

        # get first two potential components
        new_skel_score = self.get_skel_score()
        dijk_edge_score = sum([ed.get_edge_score() for ed in eligible_edge.dijk[1]])

        # last one requires more work
        dijk_turn_penalties = []
        for i, ed in enumerate(eligible_edge.dijk[1]):
            next_edge = eligible_edge.dijk[1][i+1]
            dijk_turn_penalties.append(ed.get_turn_penalty(next_edge, np.pi / 4, 0.5, 2))

        # Sort the list in descending order
        sorted_penalties = sorted(dijk_turn_penalties, reverse=True)

        # Drop the highest values
        n_dropped = 2 - eligible_edge.label
        dropped_penalties = sorted_penalties[n_dropped:]

        # calculate final potantial component
        dijk_turn_penalty_score = sum(dropped_penalties)


        return new_skel_score + dijk_edge_score - dijk_turn_penalty_score
    

    def get_eligible(self, n_tip):
        """Determines which edges can be added (i.e. dont violate the rules)"""
        # initialize dijkstra
        weights = {(i,j):None for i in range(len(self.superpoints)) for j in range(len(self.superpoints))}
        for edge in self.all_edge_collection.edges:
            start_index = self.superpoints_idx_map[edge.p_start]
            end_index = self.superpoints_idx_map[edge.p_end]
            # undirected
            print(self.all_edge_collection.predecessor_map[edge.p_start][0])
            weights[(start_index, end_index)]  =  edge.get_dijkstra_weight(predecessor=self.all_edge_collection.predecessor_map[edge.p_start][0])    
            weights[(end_index, start_index)]  =  edge.get_dijkstra_weight(predecessor=self.all_edge_collection.predecessor_map[edge.p_start][0])   
        

        eligible_edges = []
        for raw_edge in self.all_edge_collection.edges:
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

    
    def dijkstra(self, graph_weights, start, candidate_edge):
        """Use dijkstra"""
        start_idx = self.superpoints_idx_map[start]
        v = len(self.superpoints)
        visited = []
        D = {v:float('inf') for v in range(v)}
        dijk_predecessors = {v: None for v in range(v)}  # Track predecessors
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
                            dijk_predecessors[neighbor_idx] = current_point_idx

        # shortest path to edge endpoint 1
        if D[self.superpoints_idx_map[candidate_edge.p_start]] > D[self.superpoints_idx_map[candidate_edge.p_end]]:
            # loop through predecessors to find all edges
            current_pre_node = candidate_edge.p_end
            min_dist = D[self.superpoints_idx_map[candidate_edge.p_end]]
        
        # shortest path to edge endpoint 2
        else:
            current_pre_node = candidate_edge.p_start
            min_dist =  D[self.superpoints_idx_map[candidate_edge.p_start]]

        # get predecessors (Dijk in paper)
        dijk = []
        while dijk_predecessors[current_pre_node] is not None:
            # add edge that belongs to it
            predecessor_edge = self.all_edge_collection.point_edge_map[(current_pre_node, dijk_predecessors[current_pre_node])]
            dijk.append(predecessor_edge)
            current_pre_node = dijk_predecessors[current_pre_node]

        return min_dist, dijk

        
    def violates_basic_topology(self, candidate_edge):

        # if the endpoint of the edge is already the endpoint of another edge, 
        # this is wrong and should be deleted
        # by the principle of acylic outward graph
        if self.all_edge_collection.predecessor_map[candidate_edge.p_end] > 0:
            return True
        
        # the starting point of the edge has to coincide
        # with the end point of another
        # otherwise the graph is not connected
        if self.all_edge_collection.predecessor_map[candidate_edge.p_start] == 0 and not candidate_edge.p_start == self.base_node:
            return True
        
        return False
    
    def violate_label_topology(self, candidate_edge):
        predecessor =  self.all_edge_collection.predecessor_map[candidate_edge.p_start]
        successors = self.all_edge_collection.successor_map[candidate_edge.p_start]

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


    def __eq__(self, other: object) -> bool:
        return self.included_edge_collection.edges == other.included_edge_collection.edges
    
    def __hash__(self):
        return hash(tuple(self.included_edge_collection.edges))
                





    

        


        

