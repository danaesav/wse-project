import os
import pandas as pd
import matplotlib
# Set the backend first (important!)
matplotlib.use('Agg')  # Non-interactive backend that always works
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

root_dir = 'results'

name_mapping = {
    'FedAvg_b01' : 'FedAvg (b=0.1)',
    'FedAvg_b05' : 'FedAvg (b=0.5)',
    'FedAvg_b09' : 'FedAvg (b=0.9)',
    'FedAvg_uniform' : 'FedAvg (uniform)',
}

plt.figure(figsize=(10, 6))

# summary_data = []
# for subdir, _, files in os.walk(root_dir):
#     for file in files:
#         name = os.path.basename(subdir)
#         if file == 'global_eval_log.csv':
#             file_path = os.path.join(subdir, file)
#             df = pd.read_csv(file_path)
#
#             if 'round' in df.columns and 'eval_loss' in df.columns:
#                 plt.plot(df['round'], df['eval_loss'], label=name_mapping[name])
#         if file == 'final_logs.csv':
#             file_path = os.path.join(subdir, file)
#             df = pd.read_csv(file_path)
#             row = df.iloc[0].to_dict()
#             row["Model"] = name_mapping[name]
#             summary_data.append(row)
#
#
# plt.xlabel('Round')
# plt.ylabel('Eval Loss')
# plt.title('Evaluation Loss over Rounds')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('results/fedavg_eval_loss_plot.png')
# # plt.show()

# summary_df = pd.DataFrame(summary_data)
# summary_df.set_index("Model", inplace=True)
# summary_df = summary_df[[
#     "eval_loss", "eval_accuracy", "eval_f1", "total_time_seconds"
# ]]

# print(summary_df)
from utils import dirichlet_split, plot_df_language_distribution, plot_language_distribution_compact, \
    get_indexes_per_language


def get_client_datasets(beta):
    portion = 0.1
    DATA_DIR = "data"
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).sample(frac=portion, random_state=42)

    lang_to_indices = get_indexes_per_language(train_df)
    languages = list(train_df.language.unique())

    client_dfs = dirichlet_split(train_df, lang_to_indices, beta=beta, seed=42)
    plot_df_language_distribution(train_df, languages, "Total Distribution (Before client splitting)")
    plot_language_distribution_compact(client_dfs, languages, beta)

get_client_datasets(0.1)

get_client_datasets(0.5)

get_client_datasets(0.9)