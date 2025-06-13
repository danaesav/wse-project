import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils import dirichlet_split, plot_df_language_distribution, plot_language_distribution_compact, \
    get_indexes_per_language, FedAlgo


def map_name_to_label(name):
    if name.endswith('_uniform'):
        base = name.split('_')[0]
        return f"{base} (uniform)"
    elif '_b' in name:
        base, b = name.split('_b')
        return f"{base} (b=0.{b})"
    else:
        return name


def plot_combined_experiments(fed_algo: FedAlgo, results_dir="results", save=False):
    experiment_name = fed_algo.name
    if save:
        matplotlib.use('Agg')     # Non-interactive backend that always works
    else:
        matplotlib.use('Qt5Agg')  # Interactive backend for visualization
    summary_data = []
    for subdir, _, files in os.walk(results_dir):
        name = os.path.basename(subdir)
        if experiment_name not in name:
            continue
        print("Found experiment:", name)
        for file in files:
            if file == 'global_eval_log.csv':
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)

                if 'round' in df.columns and 'eval_loss' in df.columns:
                    plt.plot(df['round'], df['eval_loss'], label=map_name_to_label(name))
            if file == 'final_logs.csv':
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                row = df.iloc[0].to_dict()
                row["Model"] = map_name_to_label(name)
                summary_data.append(row)

    plt.xlabel('Round')
    plt.ylabel('Eval Loss')
    plt.title('Evaluation Loss over Rounds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(results_dir, f'{experiment_name.lower()}_eval_loss_plot.png'))
        plt.savefig('results/fedavg_eval_loss_plot.png')
    else:
        plt.show()

    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index("Model", inplace=True)
    summary_df = summary_df[[
        "eval_loss", "eval_accuracy", "eval_f1", "total_time_seconds"
    ]]

    print(summary_df)


def get_client_datasets(beta, data_dir: str = "data"):
    portion = 0.1
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv")).sample(frac=portion, random_state=42)

    lang_to_indices = get_indexes_per_language(train_df)
    languages = list(train_df.language.unique())

    client_dfs = dirichlet_split(train_df, lang_to_indices, beta=beta, seed=42)
    plot_df_language_distribution(train_df, languages, "Total Distribution (Before client splitting)")
    plot_language_distribution_compact(client_dfs, languages, beta)


# get_client_datasets(0.1)
#
# get_client_datasets(0.5)
#
# get_client_datasets(0.9)


if __name__ == '__main__':
    # Set the backend first (important!)
    plot_combined_experiments(FedAlgo.FedDisco, save=False)
