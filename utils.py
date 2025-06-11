from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from typing import List
import seaborn as sns
import matplotlib
from enum import Enum
matplotlib.use('Agg')


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


def compute_discrepancy_aware_weights(
    client_language_distribution_counts: List[List[int]],
    num_classes: int,
    a: float = 1.0,
    b: float = 0.0,
    metric: str = "l2"
) -> List[float]:
    """
    Compute client aggregation weights based on dataset sizes and internally computed discrepancy levels.

    Args:
        client_language_distribution_counts (List[List[int]]): Per-class label counts for each client.
        num_classes (int): Total number of label classes.
        a (float): Discrepancy penalty factor.
        b (float): Bias term.
        metric (str): Metric for discrepancy ("l2" or "kl").

    Returns:
        List[float]: Normalized aggregation weights for each client.
    """

    def compute_discrepancy(label_counts):
        if sum(label_counts) == 0:
            return 0.0
        dk = np.array(label_counts) / sum(label_counts)
        tc = np.full(num_classes, 1.0 / num_classes)

        if metric == "l2":
            return np.linalg.norm(dk - tc)
        elif metric == "kl":
            eps = 1e-10
            dk_safe = dk + eps
            tc_safe = tc + eps
            return np.sum(dk_safe * np.log(dk_safe / tc_safe))
        else:
            raise ValueError("Unsupported metric: choose 'l2' or 'kl'.")

    client_sizes = [sum(c) for c in client_language_distribution_counts]
    client_discrepancies = [compute_discrepancy(c) for c in client_language_distribution_counts]

    scores = [max(size - a * d + b, 0.0) for size, d in zip(client_sizes, client_discrepancies)]
    total = sum(scores)

    if total == 0:
        return [1.0 / len(client_language_distribution_counts)] * len(client_language_distribution_counts)
    return [score / total for score in scores]


# --------- Plotting ------------
def plot_df_language_distribution(df: pd.DataFrame, all_languages: list[str], title="Language Distribution"):
    all_languages_sorted = sorted(all_languages)
    df['language'] = pd.Categorical(df['language'], categories=all_languages_sorted, ordered=True)

    counts_series = df['language'].value_counts(dropna=False)
    full_counts_series = counts_series.reindex(all_languages_sorted, fill_value=0)

    language_counts = full_counts_series.reset_index()
    language_counts.columns = ['language', 'count']

    language_counts['language'] = pd.Categorical(
        language_counts['language'],
        categories=all_languages_sorted,
        ordered=True
    )
    language_counts = language_counts.sort_values('language')

    plt.figure(figsize=(8, 4))
    max_count = language_counts['count'].max()
    sns.barplot(data=language_counts, x='language', y='count', hue='language', palette='viridis', legend=False)

    for i, row in enumerate(language_counts.itertuples()):
        plt.text(i, row.count + max_count * 0.03, str(row.count), ha='center', va='bottom', fontsize=14)
    plt.ylim(0, max_count * 1.13)

    plt.title(title, fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Samples", fontsize=14)
    plt.xticks(ha='right', fontsize=14)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.savefig('distr_original')
    plt.show()


def plot_language_distribution(dataframes: list[pd.DataFrame], all_languages: list[str]):
    for client_id, client_df in enumerate(dataframes):
        plot_df_language_distribution(client_df, all_languages, "Client: " + str(client_id + 1))


def plot_language_distribution_compact(dataframes: list[pd.DataFrame], all_languages: list[str], beta,
                                       title="Language Distribution Among Clients (beta="):
    all_languages_sorted = sorted(all_languages)
    num_clients = len(dataframes)

    # Determine grid size
    rows = int(num_clients ** 0.5)
    cols = (num_clients + rows - 1) // rows  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)

    # GLOBAL TITLE
    global_title = f"{title}{beta})"
    fig.suptitle(global_title, fontsize=20, y=1.05, fontweight='bold')

    for client_id, (client_df, ax) in enumerate(zip(dataframes, axes.flatten())):
        # Prepare data
        client_df['language'] = pd.Categorical(client_df['language'],
                                               categories=all_languages_sorted,
                                               ordered=True)

        counts_series = client_df['language'].value_counts(dropna=False)
        full_counts_series = counts_series.reindex(all_languages_sorted, fill_value=0)
        language_counts = full_counts_series.reset_index()
        language_counts.columns = ['language', 'count']

        max_count = language_counts['count'].max()

        # Plot
        sns.barplot(data=language_counts, x='language', y='count',
                    hue='language', palette='viridis', legend=False, ax=ax)

        # Count labels
        for i, row in enumerate(language_counts.itertuples()):
            ax.text(i, row.count + max_count * 0.03, str(row.count), ha='center', va='bottom', fontsize=14)

        # Styling
        ax.set_ylim(0, max_count * 1.15)
        ax.set_title(f"Client {client_id + 1}", fontsize=20, fontweight='bold')
        ax.set_xlabel("Language", fontsize=18)
        ax.set_ylabel("Samples", fontsize=18)
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Turn off any extra subplots
    for i in range(num_clients, rows * cols):
        fig.delaxes(axes.flatten()[i])

    # Adjust layout to prevent overlap
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)

    # Save and show
    plt.savefig(f"distr_{int(beta * 10)}.png", bbox_inches='tight')
    plt.show()


# -------- Splitting ---------------
def get_indexes_per_language(df: pd.DataFrame):
    lang_to_indices = defaultdict(list)
    for lang, group in df.groupby('language'):
        lang_to_indices[lang].extend(group.index.tolist())
    return lang_to_indices


def uniform_split(df: pd.DataFrame, lang_to_indices, num_clients=4, seed: int = 42):
    client_indices = [[] for _ in range(num_clients)]
    client_language_distribution_counts = [[0 for _ in lang_to_indices] for _ in range(num_clients)]
    np.random.seed(seed)

    for lang_idx, (label, indices) in enumerate(lang_to_indices.items()):
        np.random.shuffle(indices)
        split_sizes = [len(indices) // num_clients] * num_clients
        for i in range(len(indices) % num_clients):
            split_sizes[i] += 1

        start = 0
        for client_id, size in enumerate(split_sizes):
            end = start + size
            assigned_indices = indices[start:end]
            client_indices[client_id].extend(assigned_indices)
            client_language_distribution_counts[client_id][lang_idx] += len(assigned_indices)
            start = end

    datasets = [df.loc[sorted(indices)] for indices in client_indices]
    return datasets, client_language_distribution_counts


def dirichlet_split(df: pd.DataFrame, lang_to_indices, beta: float, num_clients: int = 4, seed: int = None):
    rng = np.random.default_rng(seed)
    client_selected_indices = [[] for _ in range(num_clients)]
    client_language_distribution_counts = [[0 for _ in lang_to_indices] for _ in range(num_clients)]

    for lang_idx, (label, indices) in enumerate(lang_to_indices.items()):
        shuffled_label_indices = rng.permutation(indices)
        proportions = rng.dirichlet([beta] * num_clients)
        proportions = (proportions * len(indices)).astype(int)

        # Fix rounding issues to ensure total count matches
        diff = len(indices) - np.sum(proportions)
        for i in range(diff):
            proportions[i % num_clients] += 1

        current_idx_pos = 0
        for client_id, count in enumerate(proportions):
            selected_for_client = shuffled_label_indices[current_idx_pos: current_idx_pos + count]
            client_selected_indices[client_id].extend(selected_for_client)
            client_language_distribution_counts[client_id][lang_idx] += count
            current_idx_pos += count

    datasets = [df.loc[sorted(idx_list)] for idx_list in client_selected_indices]
    return datasets, client_language_distribution_counts


# -------- LABELING
def label_df(df):
    # Map stars to 0/1/2
    def map_sentiment(stars):
        if stars <= 2:
            return 0  # negative
        elif stars == 3:
            return 1  # neutral
        else:
            return 2  # positive
    df["label"] = df["stars"].apply(map_sentiment)
    return Dataset.from_pandas(df[["review_body", "label"]])


# ---------- Calculating weights
class FedAlgo(Enum):
    FedAvg = 1
    FedDisco = 2


def get_client_weights(client_datasets: list[Dataset], fed_algo: FedAlgo,
                       language_counts: list[list[int]] = None, **kwargs):
    if fed_algo == FedAlgo.FedAvg:
        total_size = sum(len(dataset) for dataset in client_datasets)
        assert total_size > 0
        return [len(dataset) / total_size for dataset in client_datasets]
    elif fed_algo == FedAlgo.FedDisco:
        if language_counts is None:
            raise ValueError("language_counts must be provided for FedDisco.")

        for i in range(1, len(client_datasets)):
            assert len(language_counts[i]) == len(language_counts[i-1]), "language_counts must have same length"

        num_classes = len(language_counts[0])
        a = kwargs.get("a", 1.0)
        b = kwargs.get("b", 0.0)
        metric = kwargs.get("metric", "l2")

        return compute_discrepancy_aware_weights(
            client_language_distribution_counts=language_counts,
            num_classes=num_classes,
            a=a,
            b=b,
            metric=metric
        )
    else:
        raise ValueError("FedAlgo must be either FedAvg or FedDisco.")
