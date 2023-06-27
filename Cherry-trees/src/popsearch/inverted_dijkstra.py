from queue import PriorityQueue

class Dijkstra:
    def __init__(self, nodes, edges) -> None:
        self.nodes = nodes
        self.node_index_map = {i:node for i,node in enumerate(nodes)}

        
        # Initialize dijkstra
        weights = {(i,j):None for i in range(len(self.nodes)) for j in range(len(self.nodes))}

        #
        for edge in edges:
            start_index = self.node_index_map[edge.p_start]
            end_index = self.node_index_map[edge.p_end]


            weights[(start_index, end_index)]  =  edge.get_dijkstra_weight(predecessor=self.all_edge_collection.predecessor_map[edge.p_start][0])    
            weights[(end_index, start_index)]  =  edge.get_dijkstra_weight(predecessor=self.all_edge_collection.predecessor_map[edge.p_start][0])   

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
