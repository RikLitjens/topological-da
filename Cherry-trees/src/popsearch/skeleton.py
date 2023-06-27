from typing import List
from popsearch.skeleton_components import Point, LabelEnum, EdgeSkeleton

import numpy as np


class Skeleton:
    def __init__(self, superpoints, all_edges, base_node) -> None:
        # define suerpoints and raw edges (not added yet but could be)
        self.superpoints: List[Point] = [Point(tuple(p)) for p in superpoints]

        # All edges that are found using the superpoint graph generation
        # These will be filtered to find the skeleton

        self.all_edges = [EdgeSkeleton(edge) for edge in all_edges]

        # Base of the tree, from here it will grow up
        self.base_node = Point(base_node, is_base=True)

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

        # Create a map from superpoint to index in the list
        # Initiated and not changed after
        # self.superpoints_idx_map = {}

        # # Creates a map from all the superpoints to its neighbouring edges
        # # Which represents the possibilities of adding an edge
        # # When one of the edges is included in the skeleton, all other edges are deleted
        # # Hence, initially full
        # self.successors_map = {}

        # # Defines the predecessor of each superpoint
        # # Is none, until assigned an included edge
        # # This is possible because predecessors are unique in the skeleton
        # # Initially all None
        # self.predecessor_map = {}

        # # Initialize maps
        # for index, point in enumerate(self.superpoints):
        #     self.superpoints_idx_map[point] = index
        #     self.successors_map[point] = []

        # Update neighbours
        for edge in self.all_edges:
            edge.point1.neighbouring_edges.append(edge)
            edge.point2.neighbouring_edges.append(edge)

    def get_skel_score(self):
        """
        Skel score is the optimization goal value
        """
        return sum(
            [
                self.get_reward(
                    edge,
                    self.predecessor_map[edge.p1],
                    self.base_node,
                )
                for edge in self.included_edges
            ]
        )

    def include_eligible_edge(self, edge: EdgeSkeleton):
        """
        Adds an edge to the final skeleton
        """

        # Include the edge
        self.included_edges.append(edge)

        # Update open and closed lists
        self.open_points.remove(edge.point1)
        self.closed_points.append(edge.point1)

        self.open_points.append(edge.point2)

        # Update point info
        edge.point1.add_outgoing_edge(edge)
        edge.point2.set_incoming_edge(edge)

        if edge.point1.incoming is None:
            raise Exception(f"{edge}, point 1 {edge.p1} has no incoming: Impossible")

        # Update edge info
        edge.predecessor = edge.point1.incoming
        edge.point1.incoming.successors.append(edge)

    def exclude_last_included_edge(self):
        """
        Throws out the last added edge
        """

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
            next_edge = eligible_edge.dijk[1][i + 1]
            dijk_turn_penalties.append(
                ed.get_turn_penalty(next_edge, np.pi / 4, 0.5, 2)
            )

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
        eligible_edges = []

        # First filter out the basic violations
        for raw_edge in self.all_edge_collection.edges:
            # check for basic topology violations
            if self.violate_basic_topology(raw_edge):
                eligible = False

            # and label violation
            if self.violate_label_topology(raw_edge):
                eligible = False

            # if elegible add to the list
            if eligible:
                eligible_edges.append(raw_edge)

        # Then use Dijkstra to fix the final constraint
        min_distance, dijk = self.dijkstra(weights, self.raw_edge_map, n_tip, raw_edge)
        raw_edge.dijk = (n_tip, dijk)  # save for possible later use
        eligible = True

        # check for first property of path existence
        if min_distance == float("inf"):
            eligible = False

        return eligible_edges

    def violates_basic_topology(self, candidate_edge):
        # if the endpoint of the edge is already the endpoint of another edge,
        # this is wrong and should be deleted
        # by the principle of acylic outward graph
        if self.included_edge_collection.predecessor_map[candidate_edge.p2] > 0:
            return True

        # the starting point of the edge has to coincide
        # with the end point of another
        # otherwise the graph is not connected
        if (
            self.included_edge_collection.predecessor_map[candidate_edge.p1] == 0
            and not candidate_edge.p1 == self.base_node
        ):
            return True

        return False

    def violate_label_topology(self, candidate_edge):
        predecessor = self.all_edge_collection.predecessor_map[candidate_edge.p1]
        successors = self.all_edge_collection.successor_map[candidate_edge.p1]

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

            if all_equal:
                return True

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
        return (
            self.included_edge_collection.edges == other.included_edge_collection.edges
        )

    def __hash__(self):
        return hash(tuple(self.included_edge_collection.edges))
