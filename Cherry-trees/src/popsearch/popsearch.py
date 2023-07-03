from popsearch.dijkstra import Dijkstra
from popsearch.pop_helpers import create_rank_dict
import random
from collections import Counter
from popsearch.skeleton import Skeleton
from popsearch.skeleton_components import Edge, EdgeSkeleton, Point


class PopSearch:
    def __init__(self, p_superpoints, raw_edges, p_tree_tips, base_node) -> None:
        self.K = 5
        self.k_rep = 3
        self.skeletons_k = []  # skeleton pop
        self.p_superpoints = p_superpoints
        self.raw_edges = raw_edges
        self.p_tree_tips = p_tree_tips
        self.base_node = tuple(base_node)
        self.iters_done = 0

        # Create a dijkstra object for each tree tip
        self.dijkstras = {}

    def do_pop_search(self):
        # start with empty skels
        self.initialize_population()

        # initialize all possible dijkstra objects
        for p_tree_tip in self.p_tree_tips:
            self.dijkstras[p_tree_tip] = self.create_dijkstra(p_tree_tip)

        # Assign a random tree tip to each skeleton
        for skel in self.skeletons_k:
            p_tree_tip = random.choice(self.p_tree_tips)
            skel.set_random_tree_tip(self.p_tree_tips, self.dijkstras)

        # todo
        for _ in range(500):
            self.create_next_gen()
            self.iters_done += 1
            if self.iters_done % 20 == 0:
                self.skeletons_k[0].plot()
                # self.skeletons_k[20].plot()
            print(f"Iteration {self.iters_done} done")

    def initialize_population(self):
        for _ in range(self.K):
            empty_skel = Skeleton(
                [p for p in self.p_superpoints],
                [Edge(ed.p1, ed.p2, ed.conf, ed.label) for ed in self.raw_edges],
                self.base_node,
            )

            self.skeletons_k.append(empty_skel)

    def create_dijkstra(self, p_tree_tip):
        # Initialize the dijkstra object
        example_skeleton = self.skeletons_k[0]  # All skeletons have the same superpoints and edges

        dijkstra_points = [Point(point.p) for point in example_skeleton.superpoints]
        dijkstra_p_to_points_map = {point.p: point for point in dijkstra_points}

        # Make new start point
        dijkstra_start = dijkstra_p_to_points_map[p_tree_tip]
        dijkstra_start.is_base = True

        # Create dijkstra object
        dijkstra: Dijkstra = Dijkstra(
            dijkstra_p_to_points_map.values(),
            [
                EdgeSkeleton(
                    Edge(edge.p1, edge.p2, edge.conf, edge.label),
                    dijkstra_p_to_points_map[edge.p1],
                    dijkstra_p_to_points_map[edge.p2],
                )
                for edge in example_skeleton.all_edges
            ],
            dijkstra_start,
        )

        # Initialize the dijkstra object
        dijkstra.dijkstra()

        return dijkstra

    def create_next_gen(self):
        weights = self.get_weights()
        candidate_skel_edge_pairs = list(weights.keys())
        probabilities = list(weights.values())

        chosen = []
        count = 0
        while len(chosen) != self.K:
            chosen += random.choices(
                candidate_skel_edge_pairs, weights=probabilities, k=self.K - len(chosen)
            )
            counts = Counter(chosen)

            # violators (occuring more than k_rep times)
            violators = [item for item, count in counts.items() if count > self.k_rep]

            # filter violaters (filter out all the items by keeping)
            chosen = [item for item in chosen if counts[item] <= self.k_rep]

            # add back violaters self.k_rep times
            for vio in violators:
                for _ in range(self.k_rep):
                    chosen.append(vio)

            count += 1

            if count == 50 and len(chosen) > 0:
                break

        # update generation
        self.skeletons_k = []
        print("Updating this generation")
        for pair in chosen:
            new_skel = pair[0].create_copy(self.p_tree_tips, self.dijkstras)
            new_skel.include_eligible_edge(pair[1], self.p_tree_tips)
            self.skeletons_k.append(new_skel)

    def get_weights(self):
        """
        Create weights as in the paper from the skeleton population
        """
        weights = {}
        ranks_skeleton = create_rank_dict(lambda skel: skel.get_skel_score(), self.skeletons_k)

        # define rank_skelly
        count = 0
        for skel in self.skeletons_k:
            rank_skeleton = ranks_skeleton[skel]
            eligible_edges = skel.get_eligible()

            ranks_edges = create_rank_dict(lambda ed: skel.get_potential(ed), eligible_edges)
            # print("edlible edges")
            # for ed in eligible_edges:
            #     print(ed, ranks_edges[ed], skel.get_potential(ed))

            if count == -1:
                skel.plot([eligible_edges])
                count += 1

            # define edge rank and weights
            for edge in eligible_edges:
                rank_edge = ranks_edges[edge]
                # add if double
                if (skel, edge) not in weights:
                    weights[(skel, edge)] = rank_skeleton * rank_edge
                else:
                    weights[(skel, edge)] += rank_skeleton * rank_edge

            # for item in weights.items():
            #     print(item)
        # rescale
        total_sum = sum(weights.values())
        rescaled_weights = {key: value / total_sum for key, value in weights.items()}
        return rescaled_weights
