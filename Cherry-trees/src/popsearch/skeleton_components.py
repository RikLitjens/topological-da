from enum import Enum
from typing import List
import numpy as np


class LabelEnum(Enum):
    TRUNK = 0
    SUPPORT = 1
    LEADER = 2
    SIDE = 3


class Point:
    def __init__(self, p: tuple) -> None:
        self.p = p

        # Only set when point is closed
        self.incoming_edge = None

        # Updated dynamically throughout the skeletonization process
        self.outgoing_edges = []

        # Neighbouring edges of the node
        self.neighbouring_edges = []

        # Whether it is a base or not
        self.is_base = False

    def add_neighbouring_edge(self, neighbouring_edge):
        self.neighbouring_edges.append(neighbouring_edge)

    def set_incoming_edge(self, incoming):
        self.incoming_edge = incoming

    def add_outgoing_edge(self, outgoing):
        self.outgoing_edges.append(outgoing)

    def remove_last_outgoing_edge(self):
        self.outgoing_edges = self.outgoing_edges[:-1]

    def update_edge_references(self, p_edge_map):
        # Update the references for all edges in the point

        # Incoming edge
        if self.incoming_edge is not None:
            # print(10 * "-")
            # print(self, id(self))
            # print(f"Updating incoming edge {self.incoming_edge}, id: {id(self.incoming_edge)}")
            # print(
            #     "point1",
            #     p_edge_map[(self.incoming_edge.p1, self.incoming_edge.p2)].point1,
            #     id(p_edge_map[(self.incoming_edge.p1, self.incoming_edge.p2)].point1),
            # )
            # print(
            #     f"p_edge_map: { p_edge_map[(self.incoming_edge.p1, self.incoming_edge.p2)], id(p_edge_map[(self.incoming_edge.p1, self.incoming_edge.p2)])}"
            # )
            # print(10 * "-")
            self.incoming_edge = p_edge_map[(self.incoming_edge.p1, self.incoming_edge.p2)]
            print("updated incoming", self.incoming_edge, id(self.incoming_edge))

        # Outgoing edges
        new_outgoing_edges = []
        for outgoing in self.outgoing_edges:
            new_outgoing_edges.append(p_edge_map[(outgoing.p1, outgoing.p2)])

        self.outgoing_edges = new_outgoing_edges

        # Neighbour edges
        new_neighbours = []
        for neighbour in self.neighbouring_edges:
            new_neighbours.append(p_edge_map[(neighbour.p1, neighbour.p2)])

        self.neighbouring_edges = new_neighbours

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.p == other.p
        return False

    def __hash__(self) -> int:
        return hash(self.p)

    def __repr__(self) -> str:
        return f"P{self.p}"

    def __str__(self) -> str:
        return self.__repr__()

    def __lt__(self, other):
        if isinstance(other, Point):
            return self.p < other.p


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
        return f"<<Edge {self.point1}-{self.point2}, conf={self.conf}, label={self.label}>>"

    def __repr__(self) -> str:
        return self.__str__()

    def get_weight(self):
        return self.length() * (1 - self.conf)

    def __lt__(self, other):
        return self.get_weight() < other.get_weight()

    def __eq__(self, other: object) -> bool:
        return (
            other is not None
            and self.p1 == other.p1
            and self.p2 == other.p2
            and self.label == other.label
        )

    def __hash__(self) -> int:
        return hash((self.p1, self.p2, self.label))


class EdgeSkeleton(Edge):
    def __init__(self, edge, point1, point2):
        super().__init__(edge.p1, edge.p2, edge.conf, edge.label)
        self.point1 = point1
        self.point2 = point2

        # not yet known at start
        self.dijk = (None, [])  # list with tip n and all edges to it from self
        self.predecessor: EdgeSkeleton = None  # Initially None, but set once skeleton is built
        self.successors: List[EdgeSkeleton] = []  # Initially None, but set when skeleton is built

    def get_reward(self):
        """
        Get the optimizer score of this edge
        Based on the confidence value
        the turn penalty
        and the grow direction
        """
        edge_score = self.get_edge_score(0.4)
        turn_penalty = (
            self.get_turn_penalty(np.pi / 4, 0.5, 2, self.predecessor)
            if not self.point1.is_base
            else 0
        )
        growth_penalty = self.get_growth_penalty(np.pi / 4, 0.4, 1)

        return edge_score - turn_penalty - growth_penalty

    def get_edge_score(self, alpha):
        return self.length() * (1 - ((1 - self.conf) / (1 - alpha)))

    def get_turn_penalty(self, theta, c_turn, p_turn, predecessor=None):
        """
        Turning edges are not wanted,
        straight is better (no offence to the lgbt)
        """
        if predecessor is None:
            predecessor = self.predecessor

        if self.label is not None and self.label != predecessor.label:
            return 0

        if self.angle_with(predecessor) <= theta:
            return 0

        return c_turn * (self.angle_with(predecessor) - theta) ** p_turn

    def get_growth_penalty(self, theta, c_grow, p_grow):
        """ "
        Returns a penalty when the label and grow direction do not match
        """
        if self.label not in [LabelEnum.LEADER, LabelEnum.SUPPORT]:
            return 0

        if self.get_delta_target() <= theta:
            return 0

        return c_grow * (self.get_delta_target() - theta) ** p_grow

    def get_delta_target(self):
        """
        Helper function that returns the angle
        of an edge to its label
        i.e. support has to be horizontal
        and leader vertical
        """
        grow_angle = self.calculate_grow_angle()
        if self.label == LabelEnum.SUPPORT:
            return grow_angle

        if self.label == LabelEnum.LEADER:
            return np.pi / 2 - grow_angle

    def get_dijkstra_weight(self, start_point):
        # cannot have a label in this step, so
        # remove to None
        label = self.label
        self.label = None
        score = (
            self.length() * (1 - self.conf) + self.get_turn_penalty(np.pi / 4, 0.5, 2)
            if self.point1 != start_point and self.point2 != start_point
            else 0
        )
        self.label = label
        return score

    def update_pred_reference(self, p_edge_map):
        # Update the reference to the predecessor
        if self.predecessor is not None:
            print(
                f"updating pred reference to {p_edge_map[(self.predecessor.p1, self.predecessor.p2)]}"
            )
            self.predecessor = p_edge_map[(self.predecessor.p1, self.predecessor.p2)]

    def update_succ_references(self, p_edge_map):
        # Update the reference to the successors
        new_successors = []
        for succ in self.successors:
            new_successors.append(p_edge_map[(succ.p1, succ.p2)])

        self.successors = new_successors
