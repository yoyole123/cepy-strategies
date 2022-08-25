import json
import pandas as pd
from uuid import uuid4


def convert_json_to_csv(json_obj: dict):
    for d in json_obj:
        d["subject_id"] = str(uuid4())
    df = pd.DataFrame(json_obj)
    for index, row in df.iterrows():
        print(row)
    df = df.melt(id_vars=['subject_id'], value_vars=["baseline_r" ,"baseline_p", "random_r", "random_p", "shortest_r", "shortest_p"])
    df.to_csv("results2.csv")


if __name__ == '__main__':
    with open("./data/2021-12-31T04_30_05_all_results.json", "r") as file:
        result_json = json.load(file)
    convert_json_to_csv(result_json)
