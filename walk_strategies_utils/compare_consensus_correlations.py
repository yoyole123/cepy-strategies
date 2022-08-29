import numpy as np
import pandas
from scipy import stats

from walk_strategies_utils.charts_creator import create_lambda_test_chart
from walks_creators.global_information_bias_tuner import GlobalInformationBiasTuner
from walks_creators.shortest_path_generator import ShortestPathGenerator
from cepy.embed_align import align
from cepy import CE

DATA_DIR = "../data"
DEFAULT_CE_PARAMETERS = {'dimensions': 30, 'walk_length': 60, 'num_walks': int(800 / 3), 'window': 3, 'p': 0.1,
                         'q': 1.6, 'permutations': 100, 'workers': 10, 'verbosity': 2, 'seed': 2021}


def compare_consensus_correlations(sc_mat_consensus: np.ndarray = None, fc_mat_consensus: np.ndarray = None,
                                   strategy: str = None, strategy_params: dict = {}, ce_params: dict = None,
                                   check_random: bool = False, return_df: bool = False):
    if strategy is None:
        strategy = "random"
    print(f"Starting to compare SC-FC correlations between regular and CE. Strategy: {strategy}. \
Strategy params: {strategy_params}")
    if sc_mat_consensus is None:
        sc_mat_consensus = np.load(f"{DATA_DIR}/sc_consensus_125.npy")
    if fc_mat_consensus is None:
        fc_mat_consensus = np.load(f"{DATA_DIR}/fc_consensus_125.npy")
    pregenerated_walks = None
    walks_generator = None
    if strategy == "shortest":
        walks_generator = ShortestPathGenerator(sc_mat_object=sc_mat_consensus, **strategy_params)
        pregenerated_walks = walks_generator.generate_walks()
    elif strategy == "stochastic":
        walks_generator = GlobalInformationBiasTuner(proximity_matrix=sc_mat_consensus, **strategy_params)
        pregenerated_walks = walks_generator.generate_walks()

    # Creating boolean array pointing the direct nodes and the lower triangle
    direct_mat = np.logical_and(sc_mat_consensus > 0, np.tri(len(sc_mat_consensus), k=-1))
    baseline_r, baseline_p = stats.pearsonr(sc_mat_consensus[direct_mat].flatten(),
                                            fc_mat_consensus[direct_mat].flatten())

    if ce_params is None:
        ce_params = DEFAULT_CE_PARAMETERS

    # Creating consensus CE for later alignment
    print("Creating CE model using consensus matrix")
    strategy_ce_p, strategy_ce_r = _get_ce_correlations(ce_params, direct_mat, fc_mat_consensus, pregenerated_walks,
                                                        sc_mat_consensus)
    random_ce_r = None
    random_ce_p = None
    if check_random is True and (strategy is not None and strategy != "random"):
        random_ce_p, random_ce_r = _get_ce_correlations(ce_params, direct_mat, fc_mat_consensus, None, sc_mat_consensus)
    print(
        f"Baseline: r={round(baseline_r, 2)}, p={round(baseline_p, 2)}, CE: r={round(strategy_ce_r, 2)} p={round(strategy_ce_p, 2)}")
    if return_df is True:
        if strategy == "shortest":
            title_row = ["baseline_r", "baseline_p", "random_r", "random_p", "shortest_r", "shortest_p"]
            result_object = pandas.DataFrame(
                [[baseline_r, baseline_p, random_ce_r, random_ce_p, strategy_ce_r, strategy_ce_p]]
                , columns=title_row)
        elif strategy == "stochastic":
            title_row = ["lambda_value", "community_bias_value", "baseline_r", "baseline_p", "lambda_r", "lambda_p"]
            result_object = pandas.DataFrame([
                [strategy_params.get("lambda_value"), strategy_params.get("community_bias_value"),
                 baseline_r, baseline_p, strategy_ce_r, strategy_ce_p]
            ], columns=title_row)
        else:
            title_row = ["baseline_r", "baseline_p", "random_r", "random_p", "shortest_r", "shortest_p"]
            result_object = pandas.DataFrame(
                [[baseline_r, baseline_p, random_ce_r, random_ce_p, strategy_ce_r, strategy_ce_p]]
            , columns=title_row)
    else:
        result_object = {
            "baseline":
                {"r": baseline_r, "p": baseline_p},
            f"ce_{strategy}":
                {"r": strategy_ce_r, "p": strategy_ce_p}
        }
        if random_ce_r is not None:
            result_object["ce_random"] = {"r": random_ce_r, "p": random_ce_p}
    return result_object


# {'baseline': {'r': 0.22941515849166347, 'p': 1.8419677239139747e-46}, 'ce_shortest': {'r': -0.14401733340293404, 'p': 5.06337803974001e-19}, 'ce_random': {'r': -0.031684713862189226, 'p': 0.05109167384171899}}
# ,baseline_r,baseline_p,random_r,random_p,shortest_r,shortest_p
# 0,0.2294151584916634,1.8419677239139747e-46,0.43493043714475627,8.848272960740988e-175,0.22827591798916883,5.249782120381298e-46

def _get_ce_correlations(ce_params, direct_mat, fc_mat_consensus, pregenerated_walks, sc_mat_consensus):
    ce_model_consensus = CE(pregenerated_walks=pregenerated_walks, **ce_params)
    ce_model_consensus.fit(sc_mat_consensus)
    consensus_aligned = align(ce_model_consensus, ce_model_consensus)
    similarity_matrix = consensus_aligned.similarity(method='cosine_similarity')
    ce_r, ce_p = stats.pearsonr(similarity_matrix[direct_mat].flatten(),
                                fc_mat_consensus[direct_mat].flatten())
    return ce_p, ce_r


if __name__ == '__main__':
    ce_params = {'dimensions': 30, 'walk_length': 60, 'num_walks': 1, 'window': 3, 'p': 0.1,
                 'q': 1.6, 'permutations': 100, 'workers': 10, 'verbosity': 2, 'seed': 2021}
    # results_df = pandas.DataFrame([["lambda_value", "community_bias_value", "baseline_r", "baseline_p", "lambda_r", "lambda_p"]])
    columns = ["lambda_value", "community_bias_value", "baseline_r", "baseline_p", "lambda_r", "lambda_p"]
    results_df = pandas.DataFrame(columns=columns)
    for community_bias_value in [50, 100]:
        for lambda_value in [10, 50]:
            strategy_params = {
                "lambda_value": lambda_value, "community_bias_value": community_bias_value,
                "walks_for_node": 1}  # For stochastic
            res = compare_consensus_correlations(strategy="stochastic", strategy_params=strategy_params,
                                                 ce_params=ce_params, check_random=True,
                                                 return_df=True)  # For stochastic
            last_row = res.tail(1)
            results_df = pandas.concat([results_df, last_row], ignore_index=True)

    # strategy_params = {"walks_for_node": 2}  # For shortest
    # res = compare_consensus_correlations(strategy="shortest", strategy_params=strategy_params, ce_params=ce_params,
    #                                      check_random=True, return_df=True)  # For shortest

    # res = compare_consensus_correlations()  # For random
    results_df.to_csv("lambda_results.csv")
    print(results_df)
    create_lambda_test_chart(csv_file_path="lambda_results.csv")
