import pandas as pd
import matplotlib.pyplot as plot
import seaborn


def create_shortest_test_chart(csv_file_path: str):
    try:
        cols_to_use = ["baseline_r", "shortest_r", "random_r"]
        df = pd.read_csv(csv_file_path, usecols=cols_to_use)[cols_to_use]

        df["subject"] = df.index
        df.set_index("subject", inplace=True)
        draw_df = df.reset_index().melt(id_vars='subject', var_name='strategy')
        ax = seaborn.lineplot(x='strategy',
                     y='value',
                     hue='subject',
                     data=draw_df)
        ax.get_legend().remove()
        ax.set_xticklabels(["Raw SC", "Shortest path CE cosine", "Random walk CE cosine"])
        plot.show()
    except Exception as e:
        print(f"Error while trying to generate plot: {type(e).__name__}: {e}")


def create_lambda_test_chart(csv_file_path: str):
    try:
        df = pd.read_csv(csv_file_path)
        x_var = "lambda_value"
        y_var = "lambda_r"
        community_bias_values = df["community_bias_value"].unique()

        # Line chart generation
        filtered_dfs = [df[df["community_bias_value"] == val] for val in community_bias_values]
        colors = ["red", "blue", "orange", "purple", "black", "yellow", "grey", "brown"]
        for d in filtered_dfs:
            color = colors.pop()
            seaborn.lineplot(x=x_var, y=y_var, data=d, color=color)
        ax = plot.gca()
        # ax.title.set_text("This is a title")
        ax.legend(title="Community Bias Value", labels=[v for v in community_bias_values])
        ax.set(xlabel="Lambda Value", ylabel="Spearman's correlation coefficient")

        # This section is used to create more tick than labels, for visual effect.
        ax.set_xticks(range(0, 1001, 50))
        labels = [str(i) for i in range(0, 1001, 100)]
        # Inserting "" between every 2 elements in labels
        result = [""] * (len(labels) * 2 - 1)
        result[0::2] = labels
        ax.set_xticklabels(result)
        plot.show()
    except Exception as e:
        print(f"Error while trying to generate plot: {type(e).__name__}: {e}")


if __name__ == '__main__':
    # create_lambda_test_chart("../lambda_value_test/data/lambda_test_results.csv")
    create_shortest_test_chart("../shortest_vs_random_test/data/shortest_vs_random_results.csv")
