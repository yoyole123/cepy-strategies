import networkx
import numpy as np
from networkx import Graph
from networkx.algorithms.shortest_paths import generic
import logging
import sys
from time import time


class ShortestPathGenerator:
    DEFAULT_WALKS_FOR_EACH_NODE = 800
    WALK_LENGTH = 20  # as in the paper

    def __init__(self, sc_mat_object: np.ndarray, walks_for_node: int = None):
        self.logger = self._set_logger()
        self.start_time = time()
        self.sc_mat_object = sc_mat_object
        self.graph_object = Graph(self.sc_mat_object)
        self.walks = []
        shape = self.sc_mat_object.shape[0]
        self.shortest_paths_matrix = np.empty(shape ** 2, dtype=object).reshape(shape, shape)
        if walks_for_node is not None:
            self.walks_for_node = walks_for_node
        else:
            self.walks_for_node = self.DEFAULT_WALKS_FOR_EACH_NODE

    def generate_walks(self, walks_for_node: int = None):
        """
        This is the main function - generate walks that are made of multiple shortest routes that are calculated,
        until the walk is big enough, according to the "WALK_LENGTH" constant which is defined above.
        :return: List of generated walks.

        Note: Code is keeping adding targets as long as current walk length is less than 20,
                but that means that the final length of each walk could end up larger than 20.
                (For example - if current length is 19 and we sampled a new destination which is 4 steps away.)
        """
        self.logger.info("Starting to generate walks")
        if walks_for_node:
            num_of_walks_for_node = walks_for_node
        else:
            num_of_walks_for_node = self.walks_for_node
        number_of_nodes = self.sc_mat_object.shape[0]
        all_nodes = np.arange(number_of_nodes)
        # Doing the entire process for each node
        for node in all_nodes:
            self.logger.debug(f"Using node {node} as source node - generating {num_of_walks_for_node} walks")
            fixed_source_node = node

            connections_left = True
            for i in range(num_of_walks_for_node):
                if connections_left is False:
                    break
                # this variable, "current_source_node", will keep changing until the entire walk is calculated.
                connections_left = True
                current_source_node = fixed_source_node
                target_nodes = list()
                target_nodes.append(fixed_source_node)
                full_path = [current_source_node]  # This will be extended with all additional nodes in this walk
                current_walk_length = len(full_path)
                while current_walk_length < self.WALK_LENGTH and connections_left is True:
                    # Calculating how many nodes are left until the full path matches "self.WALK_LENGTH"
                    number_of_nodes_to_add = self.WALK_LENGTH - len(full_path)
                    # Choosing a random destination, and excluding the nodes that were used as target before
                    try:
                        try:
                            current_destination_node = np.random.choice(np.setdiff1d(all_nodes, target_nodes))
                        except ValueError as e:
                            # In case no more valid nodes left to sample
                            self.logger.warning(
                                f"Can't create any more short walks from {current_source_node}, \
since walks were created for all connected nodes, Moving to the next source node.")
                            connections_left = False
                            continue
                        # Adding new destination to "target_nodes", so it will not be picked again
                        target_nodes.append(current_destination_node)
                        shortest_path = self._get_shortest_path(source_node=current_source_node,
                                                                destination_node=current_destination_node)
                        # Calculates the int which will be used as top of range of indices when adding new path to full_path. Note that None as index range is ignored.
                        items_to_add = number_of_nodes_to_add + 1 if len(
                            shortest_path) > number_of_nodes_to_add else None
                        # Adding new path to full_path, excluding the first node (which was already entered as the last node from last path) and making sure that the path's length is not longer than "self.WALK_LENGTH"
                        full_path.extend(shortest_path[1:items_to_add])
                        current_walk_length += len(shortest_path) - 1
                        # Setting the sampled target node as the new source node
                        current_source_node = shortest_path[-1]
                    except networkx.exception.NetworkXNoPath as err:
                        f"Problem while trying to create shortest path from node {current_source_node} to node \
{current_destination_node}: Path doesn't exist, moving on to another destination node."
                if connections_left is True:
                    parsed_full_path = [int(e) for e in full_path]
                    self.walks.append(parsed_full_path)
                    self.logger.debug(
                        f"Walk no. {i + 1} / {num_of_walks_for_node} for node no. {fixed_source_node} - Full path - Source: {full_path[0]}, Destination: {full_path[-1]}, Path: {full_path}")

        end_time = time()
        self.logger.info(
            f"Shortest paths generator finished creating a total of {self.sc_mat_object.shape[0] * self.DEFAULT_WALKS_FOR_EACH_NODE} walks after {round(end_time - self.start_time, 2)} seconds.")
        print(f"WALKS: {len(self.walks)}, {len(self.walks) / len(self.sc_mat_object)} per node")
        return self.walks

    def _get_shortest_path(self, source_node: int, destination_node: int):
        """
        Returns weighted shortest path between given source and destination nodes. If path has been calculated before,
        skipping the calculation and returning the previously calculated route.
        :param shortest_paths_matrix: Numpy matrix containing the previously calculated shortest paths,
        to avoid unnecessary re-calculations.
        :param source_node: Source node number (int)
        :param destination_node: Destination node number (int)
        :return: shortest path between given source and destination nodes
        """
        if self.shortest_paths_matrix[source_node, destination_node] is None:
            # if networkx.has_path(self.graph_object, source_node, destination_node) is False:
            #     print("OH OH", source_node, destination_node)
            shortest_path = generic.shortest_path(self.graph_object, source=source_node, target=destination_node,
                                                  weight="weight", method='dijkstra')
            self.shortest_paths_matrix[source_node, destination_node] = shortest_path
            self.shortest_paths_matrix[destination_node, source_node] = shortest_path
        else:
            shortest_path = self.shortest_paths_matrix[source_node, destination_node]
        return shortest_path

    @staticmethod
    def _set_logger():
        logger = logging.getLogger()
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


if __name__ == '__main__':
    import json
    from datetime import datetime

    # sc_mat = np.load("C:/Users/krnyo/OneDrive/Documents/לימודים/שנה ג'/מחקר מודרך גליה/code/sub1_sc_matrix.npz")['x']
    # sc_mat = np.load("C:/Users/krnyo/OneDrive/Documents/לימודים/שנה ג'/מחקר מודרך גליה/cepy-master/cepy-master/cepy/data/sc_group_matrix.npz")['x']
    sc_mat = np.load("../data/sc_consensus_125.npy")

    # Use this to generate new shortest walks file:
    shortest_paths_generator = ShortestPathGenerator(sc_mat_object=sc_mat)
    walks = shortest_paths_generator.generate_walks(walks_for_node=1)
    # current_time_string = datetime.now().strftime('%m_%d_%Y_%H_%M')
    # with open(f"../data/shortest_walks_{current_time_string}.json", "w") as file:
    #     json.dump(walks, file)

    # # If shortest walks file already exists:
    # with open("../examples/data/shortest_walks.json", "r") as file:
    #     walks = json.load(file)

    ce_model = CE(dimensions=30, walk_length=20, permutations=1,
                  num_walks=2, save_walks=True, pregenerated_walks=walks)
    # ce_model = CE(dimensions=30, walk_length=20, permutations=1,
    #               num_walks=2, save_walks=True)
    ce_model.fit(sc_mat)
    current_walk_nodes = [[int(float(j)) for j in i] for i in ce_model.walks]
    print('Shortest walk nodes indices:', current_walk_nodes[0])
