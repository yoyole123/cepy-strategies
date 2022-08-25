import logging
import sys
import numpy as np
from scipy.stats import entropy
import math
from networkx import Graph
import networkx
from tqdm import tqdm
from multiprocessing import Pool, Manager
from matplotlib import pyplot as plt


class GlobalInformationBiasTuner:
    """
    Creation of walks, for CE, that are created by using the stochastic model.
    (More info here: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006833)
    """
    DEFAULT_NUMBER_OF_WALKS = 266
    WALK_LENGTH = 60
    DEBUG_MODE = False
    MIN_THREADS = 8

    def __init__(self, proximity_matrix: np.ndarray, lambda_value: float, community_bias_value: float,
                 walks_for_node: int = DEFAULT_NUMBER_OF_WALKS, n_threads: int = 4):
        """
        Parameters
        ----------
        proximity_matrix: Numpy ndarray. Structural brain connectivity matrix.
        lambda_value: Float. Used by the stochastic model, with low value yielding random walks, and high values yielding
            walks that are made of to the shortest paths available between the current node the the next sampled target node
        community_bias_value: Float. Used by the stochastic model, with low value yielding walks that stay inside
            "community" of nodes, and high values yielding walks that tend to explore outside "community".
        walks_for_node: Int. How many walks should be created from each source node. More walks yield
            better CE results.
        n_threads: Int. How many cores should be used to create the walks, the more cores - the faster the process.
        """
        assert np.all(proximity_matrix >= 0), 'Proximity matrix cannot contain negative values'
        self.logger = self._set_logger()
        self.logger.debug(f"Initializing GlobalInformationBiasTuner...")
        self.targets_used = None
        self.proximity_matrix = self._scale_mat(proximity_matrix)
        self.distance_matrix = np.power(proximity_matrix, -1)
        self.distance_matrix = self._scale_mat(self.distance_matrix)
        self.distance_graph = Graph(self.distance_matrix)
        self.number_of_nodes = self.proximity_matrix.shape[0]
        self.all_nodes = np.arange(self.number_of_nodes)
        self.lambda_value = lambda_value
        self.community_bias_value = community_bias_value
        self.walks_for_node = walks_for_node
        self._create_shortest_distance_from_graph()
        self.sample_probabilities_matrix = self._create_sample_probabilities_matrix(self.community_bias_value)
        pool_manager = Manager()  # This allows saving to this variable by multiple processes (to save time)
        self.walks = pool_manager.list()
        self.entropies = []
        self.n_threads = n_threads

    def generate_walks(self, walks_for_node=None):
        """
        Creating walks to be used as data-set by cepy module. Orchestrating the work of "_create_chunk_of_walks"
        function and dividing the work to multiple workers by using multiprocessing.

        self.number_of_walks_per_node is used to determine number of walks to be created.

        Returns
        -------
        List of walks. (Also stored in self.walks)
        """
        if walks_for_node is None:
            walks_for_node = self.walks_for_node
        self.logger.info(f"Global Information Bias Tuner - Starting to create {walks_for_node} walks \
for each node with given values: Lambda value {self.lambda_value}, Community Bias Value {self.community_bias_value}")
        # Creating progress bar using tqdm lib, in order to visualize the process for the researcher.
        pbar = tqdm(total=len(self.all_nodes), desc='Generating walks', colour='#ffffff', file=sys.stdout)
        # Creating (walks_for_node) walks for each node.
        for node in self.all_nodes:
            self.logger.debug(f"Now creating walks with origin node {node}")
            pbar.update(1)
            # Skipping disconnected nodes (Nodes the have no connection to other nodes in self.proximity_matrix)
            if np.nansum(self.sample_probabilities_matrix[node]) == 0:
                self.logger.warning(f"Cannot start walks from node {node} because it is disconnected. \
Moving on the the next node.")
                continue
            if self.n_threads < self.MIN_THREADS or walks_for_node < 50:
                # The operation of dividing the work is also time consuming, in these cases one thread is faster
                self.logger.debug(f"self.n_threads < self.MIN_TREADS (={self.n_threads}) or \
walks_for_node < 50 (={walks_for_node}), so using only one thread")
                self._create_chunk_of_walks(chunk=list(range(walks_for_node)), node=node)
            else:
                # When multiprocessing is faster - splitting work to chunks for multiple workers
                self.logger.debug(f"Splitting work to {self.n_threads} chunks, one for each worker (processes)")
                walk_chunks = np.array_split(range(walks_for_node), self.n_threads)
                # Setting up the parameters that will be given to self._create_chunk_of_walks
                pool_data = [[c, node] for c in walk_chunks]
                # Setting up multiple workers by using multiprocessing Pool module.
                pool = Pool(self.n_threads)
                pool.starmap(self._create_chunk_of_walks, pool_data)  # Starting workers using multiprocessing Pool
                pool.close()
                pool.join()  # Waiting for all workers to finish before moving on to next node
                self.logger.debug(f"Done creating {walks_for_node} walks for node {node}")
        self.logger.info(f"Done! Created {len(self.walks)} walks.")
        pbar.close()
        return self.walks

    def _create_single_walk(self, node: int):
        """
        Creating a walk, by calculating probabilities using the stochastic model,
        taking into account lambda value and community bias value.

        Parameters
        ----------
        node: Which node to start the walk from
        save_probabilities_arrays: Should system save into memory the probability array? Was useful during dev

        Returns
        -------
        current_walk: Walk of length self.WALK_LENGTH
        """
        current_walk = [node]
        self.targets_used = []
        current_t = self._sample_target(current_node=node)
        previous_ts = [None, current_t]
        self.targets_used.append(current_t)
        while len(current_walk) < self.WALK_LENGTH:
            current_i_node = current_walk[-1]
            probabilities_list = list()
            for j in self.all_nodes:
                # Calculating probability matrix using the stochastic model
                probabilities_list.append(self._calculate_stochastic_model_walk_prob(i_node=current_i_node,
                                                                                     j_node=j, t_node=current_t,
                                                                                     lambda_value=self.lambda_value))
            probabilities_array = np.array(probabilities_list)  # Converting to Numpy array object
            probabilities_array[np.isnan(probabilities_array)] = 0  # Removing NaNs from array
            z_value = np.sum(probabilities_array)  # normalization factor
            probabilities_array = probabilities_array / z_value
            if np.sum(probabilities_array) == 0:
                raise ValueError('Lambda is too big', self.lambda_value)
            try:
                next_node = int(np.random.choice(self.all_nodes, p=probabilities_array))
            except ValueError as e:
                # This is when probabilities do not sum to 1, meaning lambda too big.
                raise ValueError(
                    f"Error caused because of lambda value being too big - {self.lambda_value} - error: {str(e)}")
            self.logger.debug(f"Next node in walk: {next_node}")
            current_walk.append(next_node)
            if next_node == current_t:
                current_t = self._sample_target(current_node=next_node, previous_target=previous_ts[0])
                previous_ts = [previous_ts[-1], current_t]
        return current_walk

    def _create_chunk_of_walks(self, chunk: list, node: int, save_probabilities_arrays: bool = False):
        """
        This function is used by multiprocessing workers, to create a chunk of walks

        Parameters
        ----------
        chunk: list of walk index that should be created (i.e. [20, 21, 22, 23...])
        node: Which node is the starting node for all walks
        save_probabilities_arrays: Should system save into memory the probability array? Was useful during dev

        Returns
        -------
        Chunk of walks

        """
        chunk_walks = []
        for n in chunk:
            self.logger.debug(f"Creating walk no. {n + 1} / {self.walks_for_node} for node {node}")
            current_walk = self._create_single_walk(node=node)
            chunk_walks.append(current_walk)
            self.logger.debug(f"Created walk {n + 1} / {self.walks_for_node} for node {node}")
        self.walks.extend(chunk_walks)
        return chunk_walks

    def _create_shortest_distance_from_graph(self):
        """Creating matrix of shortest distances between every two nodes in self.distance_graph"""
        self.shortest_distance = np.zeros(shape=(self.number_of_nodes, self.number_of_nodes))
        self.logger.debug("Calculating shortest path length for all nodes, this will take about a few seconds...")
        shortest_distances_data = [e for e in
                                   networkx.all_pairs_dijkstra_path_length(self.distance_graph, weight='weight')]
        # For each node (row) in self.distance_graph, calculate shortest distance to every other node and store result.
        for row in range(self.number_of_nodes):
            for col in shortest_distances_data[row][1].keys():
                self.shortest_distance[row][col] = shortest_distances_data[row][1][col]

    def _calculate_stochastic_model_walk_prob(self, i_node: int, j_node: int, t_node: int, lambda_value: float):
        """
        Creating probability to get fro i_node to j_node, with target of t_node.

        Parameters
        ----------
        i_node: Current node in walk
        j_node: Next possible node in walk
        t_node: The walk's target node

        Returns
        -------
        Probability (from 0 to 1) to go to j_node next.

        """
        if self.distance_graph.has_edge(i_node, j_node):
            d = self.distance_matrix[i_node, j_node]
            g = self.shortest_distance[j_node, t_node]
            inner_parenthesis = (lambda_value * (d + g)) + d
            result = math.exp(-1 * inner_parenthesis)
            return result
        else:
            return 0

    def _create_sample_probabilities_matrix(self, community_bias):
        """
        Calculating to probability to sample each node as target, using community_bias.

        Returns
        -------
        None, after saving self.sample_probabilities_matrix.

        """
        sample_probabilities_matrix = np.zeros(shape=(self.number_of_nodes, self.number_of_nodes))
        for row in range(self.number_of_nodes):
            short_dist_row = self.shortest_distance[row].copy()
            short_dist_row[(short_dist_row == np.inf) | (short_dist_row == 0)] = np.nan
            short_dist_row_inv = np.power(short_dist_row, -1)
            if community_bias == 0:
                if np.isnan(np.nanmax(short_dist_row_inv)):
                    continue
                index_of_max_value = np.nanargmax(short_dist_row_inv)
                sample_probabilities_matrix[row][index_of_max_value] = 1
                # max(short_dist_row_inv[short_dist_row_inv > 0])
            else:
                # values to probability vector:
                sample_probabilities_matrix[row] = self._softmax(short_dist_row_inv, community_bias)
        return np.nan_to_num(sample_probabilities_matrix)

    def _softmax(self, array: np.ndarray, temperature: float):
        """Using Softmax function to convert vector to probability distribution"""
        array = (array - np.nanmax(array))  # to avoid overflow
        array_exp_community_bias = np.exp(array / temperature)
        softmax_result = array_exp_community_bias / np.nansum(array_exp_community_bias)
        return softmax_result

    def _sample_target(self, current_node: int, previous_target=None) -> int:
        """Sampling new target node by using self.sample_probabilities_matrix."""
        self.logger.debug(f"Sampling a new target node. Current node: {current_node}, Prev target: {previous_target}")
        probabilities = self.sample_probabilities_matrix[current_node]
        if (previous_target is not None) & \
                (np.nansum(probabilities[np.arange(len(probabilities)) != previous_target]) > 0):
            probabilities = probabilities.copy()
            probabilities[previous_target] = 0
            probabilities = probabilities / np.sum(probabilities)
        target = int(np.random.choice(self.all_nodes, p=np.nan_to_num(probabilities)))
        self.targets_used.append(target)
        self.logger.debug(f"Sampled target: {target}")
        return target

    @staticmethod
    def _scale_mat(mat):
        mat[np.isfinite(mat)] = mat[np.isfinite(mat)] / np.max(mat[np.isfinite(mat)])
        return mat

    def plot_entropy_target(self, bias_values: float):
        """Plot entropy of target node as a function of community bias"""
        bias_values = np.sort(bias_values)
        entropies_m = np.zeros(len(bias_values))
        entropies_std = np.zeros(len(bias_values))
        for i, bias in enumerate(bias_values):
            prob_matrix = np.nan_to_num(self._create_sample_probabilities_matrix(bias))
            curr_entropies = entropy(prob_matrix, axis=1)
            entropies_m[i] = np.nanmean(curr_entropies)
            entropies_std[i] = np.nanstd(curr_entropies)
        plt.errorbar(bias_values, entropies_m, yerr=entropies_std)
        plt.xlabel('Community bias values')
        plt.ylabel('Entropy')
        plt.show()

    def plot_entropy_walk(self, lambda_values: float, sub_sample: int = 10):
        """Plot entropy of walk probability as a function of the lambda value"""
        lambda_values = np.sort(lambda_values)
        entropies_m = np.zeros(len(lambda_values))
        entropies_std = np.zeros(len(lambda_values))
        rng = np.random.default_rng()
        for lambda_i, lambda_value in enumerate(lambda_values):
            probabilities = np.zeros((sub_sample, sub_sample, self.number_of_nodes))
            for target_i, target in enumerate(rng.choice(
                    self.all_nodes, size=sub_sample, replace=False)):
                for curr_node_i, curr_node in enumerate(rng.choice(
                        self.all_nodes, size=sub_sample, replace=False)):
                    for next_node_i, next_node in enumerate(self.all_nodes):
                        probabilities[target_i, curr_node_i, next_node_i] = \
                            self._calculate_stochastic_model_walk_prob(
                                i_node=curr_node, j_node=next_node,
                                t_node=target, lambda_value=lambda_value)
            curr_entropies = entropy(probabilities, axis=-1)
            entropies_m[lambda_i] = np.nanmean(curr_entropies)
            entropies_std[lambda_i] = np.nanstd(curr_entropies)
        plt.errorbar(lambda_values, entropies_m, yerr=entropies_std)
        plt.xlabel('Lambda values')
        plt.ylabel('Sampled entropy')
        plt.show()

    def _set_logger(self):
        """Setting up a logger that will be used by this class"""
        logger = logging.getLogger()
        if (logger.hasHandlers()):
            logger.handlers.clear()
        if self.DEBUG_MODE:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        handler.setFormatter(formatter)
        if not (logger.hasHandlers()):
            logger.addHandler(handler)
        return logger


if __name__ == '__main__':
    sc_consensus = np.load(f"../data/sc_consensus_125.npy")
    global_info_bias_tuner = GlobalInformationBiasTuner(proximity_matrix=sc_consensus, lambda_value=1,
                                                        community_bias_value=1, n_threads=8)
    # global_info_bias_tuner.plot_entropy_target(np.power(np.random.uniform(low=1, high=10, size=50), 2))
    # global_info_bias_tuner.plot_entropy_walk(np.power(np.random.uniform(low=1, high=10, size=10), 3))
    global_info_bias_tuner.generate_walks()
