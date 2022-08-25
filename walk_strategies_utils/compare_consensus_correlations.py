import numpy as np
from scipy import stats
from walks_creators.global_information_bias_tuner import GlobalInformationBiasTuner
from walks_creators.shortest_path_generator import ShortestPathGenerator
from cepy.embed_align import align
from cepy import CE

DATA_DIR = "../data"
DEFAULT_CE_PARAMETERS = {'dimensions': 30, 'walk_length': 60, 'num_walks': int(800 / 3), 'window': 3, 'p': 0.1,
                         'q': 1.6, 'permutations': 100, 'workers': 10, 'verbosity': 2, 'seed': 2021}


def compare_consensus_correlations(sc_mat_consensus: np.ndarray=None, fc_mat_consensus: np.ndarray=None, strategy: str = None, strategy_params: dict = {}, ce_params: dict = None):
    if strategy is None:
        strategy = "random"
    print(f"Starting to compare SC-FC correlations between regular and CE. Strategy: {strategy}")
    if sc_mat_consensus is None:
        sc_mat_consensus = np.load(f"{DATA_DIR}/sc_consensus_125.npy")
    if fc_mat_consensus is None:
        fc_mat_consensus = np.load(f"{DATA_DIR}/fc_consensus_125.npy")
    pregenerated_walks = None
    if strategy == "shortest":
        shortest_path_generator = ShortestPathGenerator(sc_mat_object=sc_mat_consensus, **strategy_params)
        pregenerated_walks = shortest_path_generator.generate_walks()
    elif strategy == "stochastic":
        global_info_bias_tuner = GlobalInformationBiasTuner(proximity_matrix=sc_mat_consensus, **strategy_params)
        pregenerated_walks = global_info_bias_tuner.generate_walks()

    # Creating boolean array pointing the direct nodes and the lower triangle
    direct_mat = np.logical_and(sc_mat_consensus > 0, np.tri(len(sc_mat_consensus), k=-1))
    baseline_r, baseline_p = stats.pearsonr(sc_mat_consensus[direct_mat].flatten(), fc_mat_consensus[direct_mat].flatten())

    if ce_params is None:
        ce_params = DEFAULT_CE_PARAMETERS

    # Creating consensus CE for later alignment
    print("Creating CE model using consensus matrix")
    ce_model_consensus = CE(pregenerated_walks=pregenerated_walks, **ce_params)
    ce_model_consensus.fit(sc_mat_consensus)
    consensus_aligned = align(ce_model_consensus, ce_model_consensus)

    similarity_matrix = consensus_aligned.similarity(method='cosine_similarity')
    ce_r, ce_p = stats.pearsonr(similarity_matrix[direct_mat].flatten(),
                                fc_mat_consensus[direct_mat].flatten())
    print(f"Baseline: r={round(baseline_r, 2)}, p={round(baseline_p, 2)}, CE: r={round(ce_r, 2)} p={round(ce_p, 2)}")
    result_object = {
        "baseline":
            {"r": baseline_r, "p": baseline_p},
        "ce":
            {"r": ce_r, "p": ce_p}
    }
    return result_object


if __name__ == '__main__':
    # strategy_params = {"lambda_value": 50, "community_bias_value": 100, "walks_for_node": 20}  # For stochastic
    # res = compare_consensus_correlations(strategy="stochastic", strategy_params=strategy_params)  # For stochastic

    strategy_params = {"walks_for_node": 1}  # For shortest
    res = compare_consensus_correlations(strategy="shortest", strategy_params=strategy_params)  # For shortest

    # res = compare_consensus_correlations()  # For random
    print(res)
