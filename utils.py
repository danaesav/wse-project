
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import List


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





