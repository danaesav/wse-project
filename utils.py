from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from typing import List
import seaborn as sns


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
        total = sum(label_counts)
        if total == 0:
            return 0.0
        Dk = np.array(label_counts) / total
        Tc = np.full(num_classes, 1.0 / num_classes)

        if metric == "l2":
            return np.linalg.norm(Dk - Tc)
        elif metric == "kl":
            eps = 1e-10
            Dk_safe = Dk + eps
            Tc_safe = Tc + eps
            return np.sum(Dk_safe * np.log(Dk_safe / Tc_safe))
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
    sns.barplot(data=language_counts, x='language', y='count', hue='language', palette='viridis', legend=False)

    for i, row in enumerate(language_counts.itertuples()):
        plt.text(i, row.count, str(row.count), ha='center', va='bottom', fontsize=9)

    plt.title(title, fontsize=14)
    plt.xlabel("Language", fontsize=12)
    plt.ylabel("Samples", fontsize=12)
    plt.xticks(ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_language_distribution(dataframes: list[pd.DataFrame], all_languages: list[str]):
    for client_id, client_df in enumerate(dataframes):
        plot_df_language_distribution(client_df, all_languages, "Client: " + str(client_id + 1))

# -------- Splitting ---------------
def get_indexes_per_language(df: pd.DataFrame):
    lang_to_indices = defaultdict(list)
    for lang, group in df.groupby('language'):
        lang_to_indices[lang].extend(group.index.tolist())
    return lang_to_indices

def uniform_split(df: pd.DataFrame, lang_to_indices, num_clients=4, seed: int = 42):
    client_indices = [[] for _ in range(num_clients)]
    np.random.seed(seed)

    for label, indices in lang_to_indices.items():
        np.random.shuffle(indices)
        split_sizes = [len(indices) // num_clients] * num_clients
        for i in range(len(indices) % num_clients):
            split_sizes[i] += 1

        start = 0
        for client_id, size in enumerate(split_sizes):
            end = start + size
            client_indices[client_id].extend(indices[start:end])
            start = end

    datasets = [df.loc[sorted(indices)] for indices in client_indices]
    return datasets


def dirichlet_split(df: pd.DataFrame, lang_to_indices, beta: float, num_clients: int = 4, seed: int = None):
    rng = np.random.default_rng(seed)
    client_selected_indices = [[] for _ in range(num_clients)]

    for label, indices in lang_to_indices.items():
        shuffled_label_indices = rng.permutation(indices)
        proportions = rng.dirichlet([beta] * num_clients)
        proportions = (proportions * len(indices)).astype(int)

        diff = len(indices) - np.sum(proportions)
        for i in range(diff):
            proportions[i % num_clients] += 1

        current_idx_pos = 0
        for client_id, count in enumerate(proportions):
            selected_for_client = shuffled_label_indices[current_idx_pos : current_idx_pos + count]
            client_selected_indices[client_id].extend(selected_for_client)
            current_idx_pos += count

    datasets = [df.loc[sorted(idx_list)] for idx_list in client_selected_indices]
    return datasets


# -------- LABELING
def label(df):
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
from utils import compute_discrepancy_aware_weights
from enum import Enum

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
        raise NotImplementedError("FedDisco not implemented yet")
        # num_classes = kwargs.get('num_classes')
        # a = kwargs.get('a', 1.0)
        # b = kwargs.get('b', 0.0)
        # metric = kwargs.get('metric', 'l2')
        #
        # if num_classes is None:
        #     raise ValueError("num_classes must be provided for FedDisco.")
        #
        # return compute_discrepancy_aware_weights(
        #     client_language_distribution_counts=client_language_distribution_counts,
        #     num_classes=num_classes,
        #     a=a,
        #     b=b,
        #     metric=metric
        # )
        # compute_discrepancy_aware_weights()
