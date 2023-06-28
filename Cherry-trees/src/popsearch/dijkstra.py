from queue import PriorityQueue


class Dijkstra:
    def __init__(self, points, edges, start_point) -> None:
        self.points = points
        self.edges = edges
        self.start_point = start_point

        # Params filled in dijkstra

    def dijkstra(self):
        """Use dijkstra"""

        D = {point: float("inf") for point in self.points}
        D[self.start_point] = 0

        # Initialize queue with start point
        q = PriorityQueue()
        q.put((0, self.start_point))

        # Do dijkstra
        while not q.empty():
            (dist, current_point) = q.get()

            for neighbour_edge in current_point.neighbouring_edges:
                # Temp set predecessor and remove to calculate weight
                real_pred = neighbour_edge.predecessor
                neighbour_edge.predecessor = current_point.incoming
                distance = neighbour_edge.get_dijkstra_weight(self.start_point)
                neighbour_edge.predecessor = real_pred

                # Determine neighbour_point
                # (remember point1 and point2 only mean something when this is determined explicitely)
                neighbour_point = (
                    neighbour_edge.point2
                    if neighbour_edge.point2 != current_point
                    else neighbour_edge.point1
                )

                old_cost = D[neighbour_point]
                new_cost = D[current_point] + distance
                if new_cost < old_cost:
                    q.put((new_cost, neighbour_point))
                    D[neighbour_point] = new_cost

                    # Update data for the node
                    neighbour_point.set_incoming_edge(neighbour_edge)
                    current_point.add_outgoing_edge(neighbour_edge)

                    # Put points in right order and update predecessor
                    neighbour_edge.point1 = current_point
                    neighbour_edge.point2 = neighbour_point
                    neighbour_edge.predecessor = current_point.incoming
