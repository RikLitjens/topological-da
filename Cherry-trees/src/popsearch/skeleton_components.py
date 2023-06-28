from enum import Enum
from typing import List
import numpy as np


class LabelEnum(Enum):
    TRUNK = 0
    SUPPORT = 1
    LEADER = 2
    SIDE = 3


class Point:
    def __init__(self, p, is_base=False, neighbouring_edges=[]) -> None:
        self.p = p

        # Only set when point is closed
        self.incoming_edge = None

        # Updated dynamically throughout the skeletonization process
        self.outgoing_edges = []

        # Neighbouring edges of the node
        self.neighbouring_edges = neighbouring_edges

        # Whether it is a base or not
        self.is_base = is_base

    def set_neighbouring_edges(self, neighbouring_edges):
        self.neighbouring_edges = neighbouring_edges

    def set_incoming_edge(self, incoming):
        self.incoming_edge = incoming

    def add_outgoing_edge(self, outgoing):
        self.outgoing_edges.append(outgoing)

    def remove_last_outgoing_edge(self):
        self.outgoing_edges = self.outgoing_edges[:-1]

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.p == other.p
        return False

    def __hash__(self) -> int:
        return hash(self.p)


class Edge:
    def __init__(self, p1, p2, conf, label):
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.label = label
        self.conf = conf

    def angle_with(self, edge2):
        # Calculate the vectors representing the edges
        vector1 = np.array(self.p2) - np.array(self.p1)
        vector2 = np.array(edge2.p2) - np.array(edge2.p1)

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
        edge_vector = np.array(self.p2) - np.array(self.p1)

        # Calculate the dot product of the edge vector with the XZ plane vectors
        dot_product_z = np.dot(edge_vector, np.array([0, 0, 1]))
        dot_product_x = np.dot(edge_vector, np.array([1, 0, 0]))

        # Calculate the absolute values of the dot products
        abs_dot_product_z = np.abs(dot_product_z)
        abs_dot_product_x = np.abs(dot_product_x)

        # Calculate the angle in radians
        if abs_dot_product_x != 0:
            angle_radians = np.arctan(abs_dot_product_z / abs_dot_product_x)
        else:
            angle_radians = np.pi / 2

        return angle_radians

    def length(self):
        # Calculate the vector representing the edge
        edge_vector = np.array(self.p2) - np.array(self.p1)

        # Calculate the length of the edge using the Euclidean distance formula
        edge_length = np.linalg.norm(edge_vector)

        return edge_length

    def __str__(self) -> str:
        return f"<<Edge {self.p1}-{self.p2}, conf={self.conf}, label={self.label}>>"

    def __repr__(self) -> str:
        return self.__str__()

    def get_weight(self):
        return self.length() * (1 - self.conf)

    def __lt__(self, other):
        return self.get_weight() < other.get_weight()

    def __eq__(self, other: object) -> bool:
        return self.p1 == other.p1 and self.p2 == other.p2 and self.label == other.label

    def __hash__(self) -> int:
        return hash((self.p1, self.p2, self.label))


class EdgeSkeleton(Edge):
    def __init__(self, edge):
        super().__init__(edge.p1, edge.p2, edge.conf, edge.label)
        self.point1 = Point(self.p1)
        self.point2 = Point(self.p2)

        # not yet known at start
        self.dijk = (None, [])  # list with tip n and all edges to it from self
        self.predecessor: EdgeSkeleton = None  # Initially None, but set once skeleton is built
        self.successors: List[EdgeSkeleton] = []  # Initially None, but set when skeleton is built

    def get_reward(self, base_node):
        """
        Get the optimizer score of this edge
        Based on the confidence value
        the turn penalty
        and the grow direction
        """
        edge_score = self.get_edge_score(0.4)
        turn_penalty = (
            self.get_turn_penalty(self.predecessor, np.pi / 4, 0.5, 2)
            if self.p1 != base_node
            else 0
        )
        growth_penalty = self.get_growth_penalty(np.pi / 4, 0.4, 1)

        return edge_score - turn_penalty - growth_penalty

    def get_edge_score(self, alpha):
        return self.length() * (1 - ((1 - self.conf) / (1 - alpha)))

    def get_turn_penalty(self, theta, c_turn, p_turn):
        """
        Turning edges are not wanted,
        straight is better (no offence to the lgbt)
        """
        if self.label is not None and self.label != self.predecessor.label:
            return 0

        if self.angle_with(self.predecessor) <= theta:
            return 0

        return c_turn * (self.angle_with(self.predecessor) - theta) ** p_turn

    def get_growth_penalty(self, theta, c_grow, p_grow):
        """ "
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

    def get_dijkstra_weight(self, start_point):
        # cannot have a label in this step, so
        # remove to None
        label = self.label
        self.label = None
        score = (
            self.length() * (1 - self.conf)
            + self.get_turn_penalty(self.predecessor, np.pi / 4, 0.5, 2)
            if self.point1 != start_point
            else 0
        )
        self.label = label
        return score
