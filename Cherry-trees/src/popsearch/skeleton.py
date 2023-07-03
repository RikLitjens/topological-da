from typing import List
from popsearch.skeleton_components import Edge, Point, LabelEnum, EdgeSkeleton
from popsearch.dijkstra import Dijkstra

import numpy as np
import copy


class Skeleton:
    def __init__(self, superpoints, all_edges, base_node) -> None:
        # define suerpoints and raw edges (not added yet but could be)
        self.p_to_points_map = {tuple(p): Point(tuple(p)) for p in superpoints}
        self.superpoints: List[Point] = self.p_to_points_map.values()

        # All edges that are found using the superpoint graph generation
        # These will be filtered to find the skeleton
        self.all_edges = [
            EdgeSkeleton(edge, self.p_to_points_map[edge.p1], self.p_to_points_map[edge.p2])
            for edge in all_edges
        ]

        # Map from tuple p1, p2 to edge for all edges
        self.p_to_edges_map = {(edge.p1, edge.p2): edge for edge in self.all_edges}

        # Base of the tree, from here it will grow up
        if base_node in self.p_to_points_map:
            self.base_node = self.p_to_points_map[base_node]
            self.base_node.is_base = True
        else:
            self.base_node = None

        # Tree tip, the point at which the tree ends
        self.tree_tip = None
        self.dijkstra = None

        # Open points are the points at the outer edge of the skeleton
        # Here we ccan attach new edges to grow the skeleton
        self.open_points = [self.base_node]

        # Closed points are all the superpoints for which a sucessor is included in the skeleton
        self.closed_points = []

        # All the edges that are currently included in the skelly
        self.included_edges = []

        # Initialize point objects
        self.initialize_points()

    def initialize_points(self):
        """
        Create datastructures to speed up search later
        """
        # Update neighbours
        for edge in self.all_edges:
            edge.point1.add_neighbouring_edge(edge)
            edge.point2.add_neighbouring_edge(edge)

    def get_eligible(self):
        """Determines which edges can be added (i.e. dont violate the rules)"""

        # First we determine which edges from open points
        # Are in our list
        # Add all neighbour edges of open points
        initial_candidates = []
        for point in self.open_points:
            for edge in point.neighbouring_edges:
                # Do not add incoming
                if edge == point.incoming_edge:
                    continue

                # Add edge
                initial_candidates.append(edge)

                # Swap point 1 and point 2 in such a way that
                # Point 1 is the current open point (<point>)
                if edge.point2 == point:
                    edge.point1, edge.point2 = edge.point2, edge.point1
                    edge.p1, edge.p2 = edge.p2, edge.p1
                    continue

                if edge.point1 != point:
                    print(edge)
                    print(point)
                    raise Exception("These should be equal")

        print(f"Len initial {len(initial_candidates)}")

        # for ed in initial_candidates:
        #     print("initial", ed, id(ed), ed.point1, id(ed.point1), ed.point2)

        # First filter out the basic violations
        secondary_candidates = []
        for edge in initial_candidates:
            eligible = True

            # print("New edge")
            # check for basic topology violations
            if self.violates_basic_topology(edge):
                # print("violates basic topology")
                eligible = False

            # and label violation
            if self.violates_label_topology(edge):
                # print("violates label topology")
                eligible = False

            # if elegible add to the list
            if eligible:
                secondary_candidates.append(edge)

        print(f"Len secondary {len(secondary_candidates)}")

        # Define final constraint
        eligible_edges = []
        for edge in secondary_candidates:
            # Tip of the candidate edge is the target for our dijkstra
            edge.dijk = (
                self.tree_tip,
                self.dijkstra.find_path(edge.point2),
            )  # save for possible later use

            # Only eligible if there exists a path
            if len(edge.dijk[1]) > 0:
                eligible_edges.append(edge)

        # Return the edges that are eligible for addition
        print(f"Len eligible {len(eligible_edges)}")
        return eligible_edges

    def violates_basic_topology(self, candidate_edge: EdgeSkeleton):
        # The first principle of having a connected out-tree
        # Is handled by the open-point and neighbour edges mechanism

        # The second requirement is that it is acyclic
        if candidate_edge.point2 in self.open_points or candidate_edge.point2 in self.closed_points:
            return True

        return False

    def violates_label_topology(self, candidate_edge: EdgeSkeleton):
        virtual_predecessor = (
            candidate_edge.point1.incoming_edge
        )  # Virtual because it is only a candidate, not an actual edge
        virtual_successors = candidate_edge.point1.outgoing_edges + [candidate_edge]

        # label progression
        # print("label progression")
        # print(id(candidate_edge))
        # print(candidate_edge.point1, id(candidate_edge.point1))
        # print(candidate_edge, virtual_predecessor)
        if (
            not candidate_edge.point1.is_base
            and candidate_edge.label.value < virtual_predecessor.label.value
        ):
            return True

        # label linearity
        if len(virtual_successors) > 1:
            # Initially we assume all have equal label
            all_equal = True
            sample = virtual_successors[0]  # Take one random successor
            for successor in virtual_successors[1:]:
                if successor.label != sample.label:
                    # If one label is different, all_equal is false
                    all_equal = False

            # In case all labels are equal, label linearity is violated
            if all_equal:
                return True

        # Trunk_support split, check if relevant
        if (
            len(virtual_successors) > 0
            and not candidate_edge.point1.is_base
            and virtual_predecessor.label == LabelEnum.TRUNK
        ):
            # check if successors are all trunks
            # check if none of them are trunks
            # count the support succs
            all_trunk = True
            no_trunk = True
            support_count = 0

            # Check the successors
            for successor in virtual_successors:
                # If one of the labels is not a trunk, all_trunk is false
                if successor.label != LabelEnum.TRUNK:
                    all_trunk = False

                # If one of the labels is a trunk, no_trunk is false
                else:
                    no_trunk = False

                # Count the supports
                if successor.label == LabelEnum.SUPPORT:
                    support_count += 1

            # Either all successors are trunks, or none
            if not all_trunk and not no_trunk:
                return True

            # If the amount of supports is higher than 2, we violate too
            if support_count > 2:
                return True

        # otherwise
        return False

    def include_eligible_edge(self, edge: EdgeSkeleton):
        """
        Adds an edge to the final skeleton
        """

        # Make sure the edge is in the right reference frame
        edge = self.p_to_edges_map[(edge.p1, edge.p2)]

        # Include the edge
        self.included_edges.append(edge)

        # Update open and closed lists
        # print(50 * "-")
        # print("include")
        # print(id(self))
        # print(self.base_node, id(self.base_node))
        # print(self.open_points, id(self.open_points))
        # print(edge.point1, id(edge.point1))
        # print(edge.point2, id(edge.point2))
        # print("incoming")
        # print(edge.point1.incoming_edge, id(edge.point1.incoming_edge))
        # print(50 * "-")

        self.open_points.remove(edge.point1)
        self.closed_points.append(edge.point1)

        self.open_points.append(edge.point2)

        # Update point info
        edge.point1.add_outgoing_edge(edge)
        edge.point2.set_incoming_edge(edge)

        if not edge.point1.is_base and edge.point1.incoming_edge is None:
            raise Exception(f"{edge}, point 1 {edge.p1} has no incoming: Impossible")

        # Update edge info
        if not edge.point1.is_base:
            edge.predecessor = edge.point1.incoming_edge
            edge.point1.incoming_edge.successors.append(edge)

    # def exclude_last_included_edge(self):
    #     """
    #     Throws out the last added edge
    #     """

    #     # Exclude the edge
    #     last_included_edge: EdgeSkeleton = self.included_edges[-1]

    #     # Exclude edge
    #     self.included_edges = self.included_edges[:-1]

    #     # Updated open and closed lists
    #     self.open_points = self.open_points[:-1]  # Removes the endpoint of last_included_edge
    #     self.closed_points = self.closed_points[
    #         :-1
    #     ]  # Removes the start point of last_included_edge
    #     self.open_points.append(last_included_edge.point1)  # Reopen the start point

    #     # Update point info
    #     last_included_edge.point1.remove_last_outgoing_edge()
    #     last_included_edge.point2.set_incoming_edge(None)

    #     # Update edge info
    #     last_included_edge.predecessor = None
    #     last_included_edge.point1.incoming_edge.successors = (
    #         last_included_edge.point1.incoming_edge.successors[:-1]
    #     )

    def get_potential(self, eligible_edge):
        """
        Calculate the edge potential as in the paper
        """

        # get first two potential components
        new_skel_score = self.get_skel_score()
        dijk_edge_score = sum([ed.get_edge_score(0.4) for ed in eligible_edge.dijk[1]])

        # last one requires more work
        dijk_turn_penalties = [0]  # Initial turn penalty is 0 (Start of Path has no predecessor)
        for i, ed in enumerate(eligible_edge.dijk[1]):
            # Ensure index
            if i == (len(eligible_edge.dijk[1]) - 1):
                break

            # Get turn penalty
            next_edge = eligible_edge.dijk[1][i + 1]
            dijk_turn_penalties.append(
                next_edge.get_turn_penalty(np.pi / 4, 0.5, 2, predecessor=ed)
            )

        # Sort the list in descending order
        sorted_penalties = sorted(dijk_turn_penalties, reverse=True)

        # Drop the highest values
        n_dropped = 2 - eligible_edge.label.value
        dropped_penalties = sorted_penalties[n_dropped:]

        # calculate final potantial component
        dijk_turn_penalty_score = sum(dropped_penalties)

        return new_skel_score + dijk_edge_score - dijk_turn_penalty_score

    def get_skel_score(self):
        """
        Skel score is the optimization goal value
        """
        return sum([edge.get_reward() for edge in self.included_edges])

    def set_tree_tip(self, p_tree_tip, dijkstra):
        """
        Sets the tip of the skeleton
        """

        # If the tree tip is already the tree tip, do nothing
        if self.p_to_points_map[p_tree_tip] == self.tree_tip:
            return

        self.tree_tip = self.p_to_points_map[p_tree_tip]
        self.dijkstra = dijkstra

    def create_copy(self, p_tree_tip, dijkstra):
        """
        Creates a full copy of the skeleton in three steps
        1) Create point copies
        2) Create edge copies
        3) Update references within objects
        """
        new_skel = Skeleton([], [], None)

        new_skel.p_to_points_map = {}
        for p, point in self.p_to_points_map.items():
            new_point = Point(p)
            new_point.incoming_edge = point.incoming_edge  # Has to be updated later
            new_point.outgoing_edges = point.outgoing_edges  # Has to be updated later
            new_point.neighbouring_edges = point.neighbouring_edges  # Has to be updated later
            # Base point has to be updated later
            new_skel.p_to_points_map[p] = new_point
            # print("old", point, id(point), point.incoming_edge, id(point.incoming_edge))

        # Update superpoints list
        new_skel.superpoints: List[Point] = new_skel.p_to_points_map.values()

        new_skel.p_to_edges_map = {}
        for edge in self.all_edges:
            new_edge = EdgeSkeleton(
                Edge(edge.p1, edge.p2, edge.conf, edge.label),
                new_skel.p_to_points_map[edge.p1],
                new_skel.p_to_points_map[edge.p2],
            )
            new_edge.predecessor = edge.predecessor  # Has to be updated later
            new_edge.successors = edge.successors  # Has to be updated later
            new_skel.p_to_edges_map[(edge.p1, edge.p2)] = new_edge
            # predecessors and successors have to be updated later

        new_skel.all_edges = new_skel.p_to_edges_map.values()

        # Base of the tree, from here it will grow up
        new_skel.base_node = new_skel.p_to_points_map[self.base_node.p]
        new_skel.base_node.is_base = True

        # Update edge references in point
        for point in new_skel.superpoints:
            point.update_edge_references(new_skel.p_to_edges_map)
            # print("new", point, id(point), "incoming", point.incoming_edge, id(point.incoming_edge))

        # Update pre- and succesors
        for edge in new_skel.all_edges:
            edge.update_pred_reference(new_skel.p_to_edges_map)
            edge.update_succ_references(new_skel.p_to_edges_map)
            # print(edge, id(edge))

        # Update open points
        new_skel.open_points = [new_skel.p_to_points_map[point.p] for point in self.open_points]

        # Update closed points
        new_skel.closed_points = [new_skel.p_to_points_map[point.p] for point in self.closed_points]

        # Update included edges
        new_skel.included_edges = [
            new_skel.p_to_edges_map[(edge.p1, edge.p2)] for edge in self.included_edges
        ]

        # Set Tree tip and initialize dijkstra
        new_skel.set_tree_tip(p_tree_tip, dijkstra)

        # for ed1, ed2 in zip(self.included_edges, new_skel.included_edges):
        #     print(10 * "-")
        #     print(ed1, ed2)
        #     print(ed1.predecessor, ed2.predecessor)
        #     print(new_skel.base_node, new_skel.base_node.is_base)
        #     print(ed1.point1, id(ed1.point1), ed2.point1, id(ed2.point1))
        #     print(10 * "-")

        return new_skel

    def __eq__(self, other) -> bool:
        return self.included_edges == other.included_edges

    def __hash__(self):
        return hash(id(self))
