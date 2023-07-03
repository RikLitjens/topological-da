from queue import PriorityQueue
import time


class Dijkstra:
    def __init__(self, points, edges, start_point) -> None:
        self.points = points
        self.edges = edges
        self.start_point = start_point

        self.p_point_map = {}

        # Initalize neighbours within the dijkstra environment
        for edge in self.edges:
            edge.point1.add_neighbouring_edge(edge)
            edge.point2.add_neighbouring_edge(edge)

    def dijkstra(self):
        """Use dijkstra"""

        # print(f"Starting Dijkstra")
        start_time = time.time()
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
                neighbour_edge.predecessor = current_point.incoming_edge
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

                    # Put points in right order and update predecessor
                    neighbour_edge.point1 = current_point
                    neighbour_edge.point2 = neighbour_point
                    neighbour_edge.p1 = current_point.p
                    neighbour_edge.p2 = neighbour_point.p
                    neighbour_edge.predecessor = current_point.incoming_edge

                    # Update data for the node
                    neighbour_point.set_incoming_edge(neighbour_edge)
                    current_point.add_outgoing_edge(neighbour_edge)

                    # Update map of found points
                    self.p_point_map[neighbour_point.p] = neighbour_point

        # print(f"Dijkstra completed in {time.time() - start_time} secs")
        # print(f"Found {len(self.p_point_map)} points")

    def find_path(self, target_point):
        """Return the shortest path"""
        edges_path = []

        # If it is not included in the map
        # Dijkstra has not reached this target
        if target_point.p not in self.p_point_map:
            return edges_path

        # Convert the target point to the local Dijkstra point
        count = 0
        current_incoming = self.p_point_map[target_point.p].incoming_edge
        while current_incoming is not None:
            edges_path.append(current_incoming)
            current_incoming = current_incoming.point1.incoming_edge
            if count == 10_000:
                print("Infinite loop detected")

        return edges_path
