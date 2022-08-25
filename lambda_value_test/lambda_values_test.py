import os
import numpy as np
import pandas as pd
from scipy import stats

from walks_creators.global_information_bias_tuner import GlobalInformationBiasTuner
from ce import CE
from embed_align import align


DATA_DIR = "../data"
SAVED_CE_PATH = "saved_ce"
RESULTS_FILE_PATH = f"test_results.csv"
if not os.path.exists(SAVED_CE_PATH):
    os.mkdir(SAVED_CE_PATH)

CE_PARAMETERS = {'dimensions': 30, 'walk_length': 60, 'num_walks': int(800 / 3), 'window': 3, 'p': 0.1,
                 'q': 1.6, 'permutations': 100, 'workers': 10, 'verbosity': 2, 'seed': 2021}
# COMMUNITY_BIAS_VALUES = [pow(10, i) * 0.01 for i in range(5)]  # 0.01 -> 100
COMMUNITY_BIAS_VALUES = [0.5, 1, 10, 100]
LAMBDA_VALUES = np.load("lambda_values_for_test.npy")  # <- To have it consistent, if it all fails midway.
# LAMBDA_VALUES = np.power(np.random.uniform(low=1, high=10, size=1000), 3)  # <- this is the way it was generated


def _create_comparison_metadata_dict(file_couples: list) -> list:
    metadata_dicts = []
    for file_couple in file_couples:
        sc_mat = np.load(file_couple[0])
        fc_mat = np.load(file_couple[1])
        comparison_metadata = {"description": "consensus", "SC_matrix": sc_mat, "FC_matrix": fc_mat}
        metadata_dicts.append(comparison_metadata)
    return metadata_dicts


def _save_results_to_csv(all_results):
    all_results_df_for_graph = pd.DataFrame(data=all_results,
                                            columns=["lambda_value", "community_bias_value", "baseline_r", "baseline_p",
                                                     "lambda_r", "lambda_p"])
    all_results_df_for_graph.to_csv(RESULTS_FILE_PATH, encoding="utf-8")


def run_test():
    if os.path.exists(RESULTS_FILE_PATH):
        # This is to load results that were already collected.
        df = pd.read_csv(RESULTS_FILE_PATH)
        all_results = [df.loc[i, :].values.tolist()[1:] for i in range(len(df))]
    else:
        all_results = []
    matrices_to_compare = []
    consensus_files_to_compare = [(f"{DATA_DIR}/sc_consensus_125.npy", f"{DATA_DIR}/fc_consensus_125.npy")]
    # This section is used to load the consensus matrices from the files and prepare them for comparison
    metadata_dicts = _create_comparison_metadata_dict(consensus_files_to_compare)
    matrices_to_compare.extend(metadata_dicts)
    consensus_aligned_lambda = None
    try:
        for community_bias_value in COMMUNITY_BIAS_VALUES:
            print(f"Now working with community bias value of {community_bias_value}")
            for lambda_value in LAMBDA_VALUES:
                results_already_exist = len([r for r in all_results if round(r[0], 4) == round(lambda_value, 4) and
                                             r[1] == community_bias_value]) > 0
                if results_already_exist:
                    print(f"Results for community bias of {community_bias_value} and lambda value of {lambda_value} \
already exist. Moving on...")
                    continue
                print(f"Now working lambda value of {lambda_value}")
                for current_comparison_metadata in matrices_to_compare:
                    try:
                        print(f"Now comparing SC to FC in {current_comparison_metadata['description']}")
                        sc_mat = current_comparison_metadata["SC_matrix"]
                        direct_mat = np.logical_and(sc_mat > 0, np.tri(len(sc_mat), k=-1))  # boolean array pointing the direct nodes and the lower triangle
                        fc_mat = current_comparison_metadata["FC_matrix"]

                        baseline_r, baseline_p = stats.pearsonr(sc_mat[direct_mat].flatten(), fc_mat[direct_mat].flatten())

                        # Creating CE model for stochastic model walk strategy
                        global_info_bias_tuner = GlobalInformationBiasTuner(proximity_matrix=sc_mat,
                                                                            lambda_value=lambda_value,
                                                                            community_bias_value=community_bias_value,
                                                                            walks_for_node=CE_PARAMETERS["num_walks"],
                                                                            n_threads=8)
                        walks = global_info_bias_tuner.generate_walks()
                        ce_model_lambda = CE(pregenerated_walks=walks, **CE_PARAMETERS)
                        ce_model_lambda.fit(sc_mat)
                        if current_comparison_metadata["description"] == "consensus":
                            consensus_aligned_lambda = align(ce_model_lambda, ce_model_lambda)
                        ce_model_lambda_aligned = align(consensus_aligned_lambda, ce_model_lambda)
                        # GRAPHS: Saves aligned lambda CE model to file
                        ce_model_lambda_aligned.pickle_model(
                            path=os.path.join("saved_ce", f"{current_comparison_metadata['description']}\
_lambda_{lambda_value}_community_{community_bias_value}_ce.pkl"))
                        lambda_similarity_matrix = ce_model_lambda_aligned.similarity(method='cosine_similarity')
                        lambda_r, lambda_p = stats.pearsonr(lambda_similarity_matrix[direct_mat].flatten(),
                                                            fc_mat[direct_mat].flatten())

                        print(f"Result for matrix {current_comparison_metadata['description']}: Baseline: r={baseline_r}, p={baseline_p}, lambda {lambda_value} and community bias {community_bias_value}: r={round(lambda_r, 2)} p={round(lambda_p, 2)}")
                        all_results.append([lambda_value, community_bias_value, baseline_r, baseline_p, lambda_r, lambda_p])
                        _save_results_to_csv(all_results)

                    except Exception as err:
                        print(f"Error - {err} - ignoring current comparison ({current_comparison_metadata['description']}) and moving on...")
    finally:  # No matter what, save what you have so far
        _save_results_to_csv(all_results)


if __name__ == '__main__':
    run_test()
