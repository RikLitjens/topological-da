from popsearch.pop_helpers import create_rank_dict
import random
from collections import Counter
from popsearch.skeleton import Skeleton
from popsearch.skeleton_components import Edge
import copy


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

    def do_pop_search(self):
        # start with empty skels
        self.initialize_population()

        # todo
        for _ in range(500):
            self.create_next_gen()
            self.iters_done += 1
            print(f"Iteration {self.iters_done} done")

    def initialize_population(self):
        for _ in range(self.K):
            empty_skel = Skeleton(
                [p for p in self.p_superpoints],
                [Edge(ed.p1, ed.p2, ed.conf, ed.label) for ed in self.raw_edges],
                self.base_node,
            )

            empty_skel.set_tree_tip(random.choice(self.p_tree_tips))
            self.skeletons_k.append(empty_skel)

    def create_next_gen(self):
        weights = self.get_weights()
        candidate_skel_edge_pairs = list(weights.keys())
        probabilities = list(weights.values())

        chosen = []
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

        # update generation
        self.skeletons_k = []
        for candidate_pair in chosen:
            print("Updating this generation")
            new_skel = candidate_pair[0].create_copy(random.choice(self.p_tree_tips))
            new_skel.include_eligible_edge(candidate_pair[1])
            self.skeletons_k.append(new_skel)

    def get_weights(self):
        """
        Create weights as in the paper from the skeleton population
        """
        weights = {}
        ranks_skeleton = create_rank_dict(lambda skel: skel.get_skel_score(), self.skeletons_k)

        # define rank_skelly
        for skel in self.skeletons_k:
            rank_skeleton = ranks_skeleton[skel]
            p_tip = random.choice(self.p_tree_tips)
            eligible_edges = skel.get_eligible()
            ranks_edges = create_rank_dict(lambda ed: skel.get_potential(ed), eligible_edges)

            # define edge rank and weights
            for edge in eligible_edges:
                rank_edge = ranks_edges[edge]
                # add if double
                if (skel, edge) not in weights:
                    weights[(skel, edge)] = rank_skeleton * rank_edge
                else:
                    weights[(skel, edge)] += rank_skeleton * rank_edge

        # rescale
        total_sum = sum(weights.values())
        rescaled_weights = {key: value / total_sum for key, value in weights.items()}
        return rescaled_weights
